# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import csv
import copy
import os, sys
import itertools
import numpy as np
from tqdm import tqdm

from House3D.objrender import Vec3

import pdb


class House3DUtils():
    def __init__(
            self,
            env,
            rotation_sensitivity=9,
            move_sensitivity=0.5,
            build_graph=False,
            graph_dir='/path/to/3d-graphs',
            target_obj_conn_map_dir='/path/to/target_obj_connmaps',
            debug=True,
            load_semantic_classes=True,
            collision_reward=0.0,
            success_reward=1.0,
            dist_reward_scale=0.005,
            seeing_rwd=False):
        self.env = env
        self.debug = debug

        self.rotation_sensitivity = rotation_sensitivity
        self.move_sensitivity = move_sensitivity

        self.angles = [x for x in range(-180, 180, self.rotation_sensitivity)]
        self.angle_strings = {1: 'right', -1: 'left'}

        self.dirs, self.angle_map = self.calibrate_steps(reset=True)
        self.move_multiplier = self.move_sensitivity / np.array([np.abs(x).sum() for x in self.dirs]).mean()

        self.graph_dir = graph_dir
        self.graph = None

        self.target_obj_conn_map_dir = target_obj_conn_map_dir

        if build_graph == True:
            if os.path.exists(
                    os.path.join(graph_dir,
                                 self.env.house.house['id'] + '.pkl')):
                self.load_graph(
                    os.path.join(graph_dir,
                                 self.env.house.house['id'] + '.pkl'))
            else:
                self.build_graph(
                    save_path=os.path.join(
                        graph_dir, self.env.house.house['id'] + '.pkl'))

        self.rooms, self.objects = self._parse()

        self.collision_reward = collision_reward
        self.success_reward = success_reward
        self.dist_reward_scale = dist_reward_scale
        self.seeing_rwd = seeing_rwd

        if load_semantic_classes == True:
            self._load_semantic_classes()

    # Shortest paths are computed in 1000 x 1000 grid coordinates.
    # One step in the SUNCG continuous coordinate system however, can be
    # multiple grids in the grid coordinate system (since turns aren't 90 deg).
    # So even though the grid shortest path is fine-grained,
    # an equivalent best-fit path in SUNCG continuous coordinates
    # has to be computed by simulating steps. Sucks, but yeah.
    #
    # For now, we first explicitly calibrate how many steps in the gridworld
    # correspond to one step in continuous world, across all directions
    def calibrate_steps(self, reset=True):
        mults, angle_map = [], {}

        cx, cy = self.env.house.to_coor(50, 50)
        if reset == True:
            self.env.reset(x=cx, y=cy)

        for i in range(len(self.angles)):
            yaw = self.angles[i]

            self.env.cam.yaw = yaw
            self.env.cam.updateDirection()

            x1, y1 = self.env.house.to_grid(self.env.cam.pos.x,
                                            self.env.cam.pos.z)

            pos = self.env.cam.pos
            pos = pos + self.env.cam.front * self.move_sensitivity

            x2, y2 = self.env.house.to_grid(pos.x, pos.z)

            mult = np.array([x2, y2]) - np.array([x1, y1])
            mult = (mult[0], mult[1])

            angle_map[mult] = yaw
            mults.append(mult)

        return mults, angle_map

    # 0: forward
    # 1: left
    # 2: right
    # 3: stop
    #
    # returns observation, reward, done, info
    def step(self, action, step_reward=False):
        if action not in [0, 1, 2, 3]:
            raise IndexError

        if step_reward == True:
            pos = self.env.cam.pos
            x1, y1 = self.env.house.to_grid(self.env.cam.pos.x, self.env.cam.pos.z)
            init_target_dist = self.env.house.connMap[x1, y1]

        reward = 0
        done = False

        if action == 0:
            mv = self.env.move_forward(
                dist_fwd=self.move_sensitivity, dist_hor=0)
            obs = self.env.render()
            if mv == False:  # collision
                reward -= self.collision_reward
            elif mv != False and step_reward == True:
                # evaluate connMap dist here
                x2, y2 = self.env.house.to_grid(self.env.cam.pos.x,
                                                self.env.cam.pos.z)
                final_target_dist = self.env.house.connMap[x2, y2]
                reward += self.dist_reward_scale * ((init_target_dist - final_target_dist) / np.abs(
                    self.dirs[self.angles.index(self.env.cam.yaw % 180)]).sum())

        elif action == 1:
            self.env.rotate(-self.rotation_sensitivity)
            obs = self.env.render()

        elif action == 2:
            self.env.rotate(self.rotation_sensitivity)
            obs = self.env.render()

        elif action == 3:
            done = True
            obs = self.env.render()

        return obs, reward, done

    # pos: [x, y, z, yaw], or objrender.Vec3
    def get_dist_to_target(self, pos):
        if isinstance(pos, Vec3) == True:
            x, y = self.env.house.to_grid(pos.x, pos.z)
        else:
            x, y = self.env.house.to_grid(pos[0], pos[2])
        dist = self.env.house.connMap[x, y]
        return self.move_multiplier * dist

    def is_inside_room(self, pos, room):
        if isinstance(pos, Vec3) == True:
            x = pos.x
            y = pos.z
        else:
            x = pos[0]
            y = pos[2]
        if x >= room['bbox']['min'][0] and x <= room['bbox']['max'][0] and \
            y >= room['bbox']['min'][2] and y <= room['bbox']['max'][2]:
            return True
        return False

    # takes 200-300 seconds(!) when rotation_sensitivity == 9
    def build_graph(self, save_path=None):
        import time
        start_time = time.time()

        collide_res = self.env.house.n_row

        from dijkstar import Graph

        visit = dict()
        self.graph = Graph()

        self.mock_obs_map = np.zeros(
            (collide_res + 1, collide_res + 1), dtype=np.uint8)
        self.mock_obs_map[np.where(self.env.house.connMap == -1)] = 1

        for x in range(collide_res + 1):
            for y in range(collide_res + 1):
                pos = (x, y)
                if self.env.house.canMove(x, y) and pos not in visit:
                    que = [pos]
                    visit[pos] = True
                    ptr = 0
                    while ptr < len(que):
                        cx, cy = que[ptr]
                        ptr += 1

                        # add all angles for (cx, cy) here
                        # connect first and last
                        for ang in range(len(self.angles) - 1):
                            self.graph.add_edge((cx, cy, self.angles[ang]),
                                                (cx, cy, self.angles[ang + 1]),
                                                {
                                                    'cost': 1
                                                })
                            self.graph.add_edge((cx, cy, self.angles[ang + 1]),
                                                (cx, cy, self.angles[ang]), {
                                                    'cost': 1
                                                })
                        self.graph.add_edge((cx, cy, self.angles[-1]),
                                            (cx, cy, self.angles[0]), {
                                                'cost': 1
                                            })
                        self.graph.add_edge((cx, cy, self.angles[0]),
                                            (cx, cy, self.angles[-1]), {
                                                'cost': 1
                                            })

                        for deti in range(len(self.dirs)):
                            det = self.dirs[deti]
                            tx, ty = cx + det[0], cy + det[1]
                            if (self.env.house.inside(tx, ty) and
                                    self.mock_obs_map[min(cx, tx):max(cx, tx)+1,
                                                      min(cy, ty):max(cy, ty)+1].sum() == 0):
                                # make changes here to add edges for angle increments as well
                                #
                                # cost = 1 from one angle to the next,
                                # and connect first and last
                                # this would be for different angles for same tx, ty
                                #
                                # then there would be connections for same angle
                                # and from (cx, cy) to (tx, ty)
                                self.graph.add_edge(
                                    (cx, cy, self.angle_map[self.dirs[deti]]),
                                    (tx, ty, self.angle_map[self.dirs[deti]]),
                                    {
                                        'cost': 1
                                    })
                                tp = (tx, ty)
                                if tp not in visit:
                                    visit[tp] = True
                                    que.append(tp)

        if self.debug == True:
            print("--- %s seconds to build the graph ---" %
                  (time.time() - start_time))

        if save_path != None:
            start_time = time.time()

            print("saving graph to %s" % (save_path))
            self.graph.dump(save_path)

            if self.debug == True:
                print("--- %s seconds to save the graph ---" %
                      (time.time() - start_time))

    def load_graph(self, path):
        import time
        start_time = time.time()

        from dijkstar import Graph

        self.graph = Graph()
        self.graph.load(path)

        if self.debug == True:
            print("--- %s seconds to load the graph ---" %
                  (time.time() - start_time))

    # takes 1-5 seconds when rotation_sensitivity == 9
    def compute_shortest_path(self, source, target, graph=None):
        from dijkstar import find_path

        if graph == None:
            if self.graph == None:
                if os.path.exists(
                        os.path.join(self.graph_dir,
                                     self.env.house.house['id'] + '.pkl')):
                    self.load_graph(
                        os.path.join(self.graph_dir,
                                     self.env.house.house['id'] + '.pkl'))
                else:
                    self.build_graph(
                        save_path=os.path.join(
                            graph_dir, self.env.house.house['id'] + '.pkl'))
            graph = self.graph

        cost_func = lambda u, v, e, prev_e: e['cost']
        shortest_path = find_path(graph, source, target, cost_func=cost_func)

        return shortest_path

    def fit_grid_path_to_suncg(self, nodes, init_yaw=None, back_skip=2):

        # don't mess with the originals
        nodes = copy.deepcopy(nodes)

        # set initial position
        x, y = self.env.house.to_coor(nodes[0][0], nodes[0][1], True)
        x, y = x.astype(np.float32).item(), y.astype(np.float32).item()

        self.env.cam.pos.x, self.env.cam.pos.y, self.env.cam.pos.z = x, self.env.house.robotHei, y
        if init_yaw == None:
            self.env.cam.yaw = np.random.choice(self.angles)
        else:
            self.env.cam.yaw = init_yaw
        self.env.cam.updateDirection()

        pos_queue, action_queue = [], []

        current_pos = self._vec_to_array(self.env.cam.pos, self.env.cam.yaw)
        pos_queue = pos_queue + [current_pos]

        ptr = 0

        while ptr < len(nodes) - 1:
            turned = False

            # target rotation
            target_yaw = self.angle_map[tuple(
                np.array(nodes[ptr]) - np.array(nodes[ptr + 1]))]

            # turn
            if target_yaw != current_pos[3]:
                p_q, a_q = self.get_rotate_steps(current_pos, target_yaw)

                pos_queue = pos_queue + p_q
                action_queue = action_queue + a_q

                self.env.cam.yaw = target_yaw
                self.env.cam.updateDirection()

                turned = True
                current_pos = self._vec_to_array(self.env.cam.pos,
                                                 self.env.cam.yaw)

            # move
            cx, cz = self.env.house.to_coor(nodes[ptr + 1][0],
                                            nodes[ptr + 1][1], True)

            # if collision, find another sub-path, and delete that edge
            if self.env.move(cx, cz) == False:
                if nodes[ptr + 1] in self.graph[nodes[ptr]]:
                    del self.graph[nodes[ptr]][nodes[ptr + 1]]
                    print('deleted', nodes[ptr], nodes[ptr + 1])

                # delete the turns
                if turned == True:
                    pos_queue = pos_queue[:-len(p_q)]
                    action_queue = action_queue[:-len(a_q)]

                if back_skip != 0:
                    pos_queue = pos_queue[:-back_skip]
                    action_queue = action_queue[:-back_skip]

                dest_ptr = ptr + 1
                ptr = ptr - back_skip

                sub_shortest_path = self.compute_shortest_path(
                    nodes[ptr], nodes[dest_ptr])
                nodes = nodes[:ptr] + sub_shortest_path.nodes + nodes[dest_ptr
                                                                      + 1:]

                current_pos = pos_queue[-1]
            else:
                # this is the new position the agent moved to
                current_pos = self._vec_to_array(self.env.cam.pos,
                                                 self.env.cam.yaw)

                assert current_pos[3] == pos_queue[-1][3] and (
                    current_pos[0] != pos_queue[-1][0]
                    or current_pos[2] != pos_queue[-1][2])

                pos_queue = pos_queue + [current_pos]
                action_queue = action_queue + ['fwd']

                ptr = ptr + 1

        action_queue.append('stop')

        return pos_queue, action_queue

    # pos contains [x, y, z, yaw]
    # given a position and target yaw, this function
    # computes actions needed to turn there
    def get_rotate_steps(self, pos, target_yaw):

        direction = np.random.choice([1, -1])

        cur_yaw = pos[-1]
        ptr = self.angles.index(cur_yaw)
        pos_queue, action_queue = [], []

        while cur_yaw != target_yaw:
            if len(pos_queue) == len(self.angles) // 2:
                # reset
                direction = direction * -1
                cur_yaw = pos[-1]
                ptr = self.angles.index(cur_yaw)
                pos_queue, action_queue = [], []

            ptr = (ptr + direction) % len(self.angles)
            cur_yaw = self.angles[ptr]

            pos_queue.append([pos[0], pos[1], pos[2], self.angles[ptr]])
            action_queue.append(self.angle_strings[direction])

        return pos_queue, action_queue

    def _vec_to_array(self, pos, yaw):
        return [pos.x, pos.y, pos.z, yaw]

    # render images from camera position queue
    def render_images_from_pos_queue(self,
                                     pos_queue=[],
                                     img_dir='tmp/images',
                                     actions=None,
                                     values=None,
                                     rewards=None):
        if len(pos_queue) == 0:
            return False

        action_map = {0: 'FRWD', 1: 'LEFT', 2: 'RGHT', 3: 'STOP'}

        import scipy.misc

        sgx, sgy = self.env.house.to_grid(pos_queue[0][0], pos_queue[0][2])
        tgx, tgy = self.env.house.to_grid(pos_queue[-1][0], pos_queue[-1][2])

        for i in range(len(pos_queue)):
            # set position
            p = pos_queue[i]
            self.env.reset(x=p[0], y=p[2], yaw=p[3])

            # save image
            image = np.array(self.env.render(), copy=False)

            # put some text
            text = "[%02d]" % (i + 1)

            if actions != None and i < len(actions):
                text += "[%s]" % action_map[actions[i]]

            if values != None and i < len(values):
                text += "[V%.03f]" % values[i]

            if rewards != None and i > 0 and i <= len(rewards):
                text += "[R%.03f]" % rewards[i - 1]

            image = cv2.putText(
                img=np.copy(image),
                text=text,
                org=(20, 30),
                fontFace=3,
                fontScale=0.4,
                color=(255, 255, 255),
                thickness=1)

            scipy.misc.toimage(image).save(
                '%s/%s_%04d_%04d_%04d_%04d_%05d_%05d.jpg' %
                (img_dir, self.env.house.house['id'], sgx, sgy, tgx, tgy,
                 i + 1, len(pos_queue)))

        return True

    # render video from camera position queue
    #
    # NOTE: call `render_images_from_pos_queue` before calling this
    def render_video_from_pos_queue(self,
                                    pos_queue=[],
                                    img_dir='tmp/images',
                                    vid_dir='tmp/videos',
                                    fps=[5],
                                    tag_name='piano'):
        if len(pos_queue) == 0:
            return False

        import subprocess

        sgx, sgy = self.env.house.to_grid(pos_queue[0][0], pos_queue[0][2])
        tgx, tgy = self.env.house.to_grid(pos_queue[-1][0], pos_queue[-1][2])

        for fp in fps:
            subprocess.Popen([
                '/srv/share/abhshkdz/local/bin/ffmpeg', '-f', 'image2', '-r',
                str(fp), '-i',
                '%s/%s_%04d_%04d_%04d_%04d' %
                (img_dir, self.env.house.house['id'], sgx, sgy, tgx, tgy) +
                '_%05d_' + '%05d.jpg' % (len(pos_queue)), '-vcodec', 'libx264',
                '-crf', '25', '-y',
                '%s/%s_%04d_%04d_%s_%04d_%04d_%d.mp4' %
                (vid_dir, self.env.house.house['id'], sgx, sgy, tag_name, tgx,
                 tgy, fp)
            ])

            if self.debug == True:
                print('Rendered video to ' +
                      '%s/%s_%04d_%04d_%s_%04d_%04d_%d.mp4' %
                      (vid_dir, self.env.house.house['id'], sgx, sgy, tag_name,
                       tgx, tgy, fp))

        return True

    # Go over all nodes of house environment and accumulate objects room-wise.
    def _parse(self, levelsToExplore=[0]):
        rooms, objects = [], {}
        data = self.env.house.house

        modelCategoryMapping = {}

        import csv
        csvFile = csv.reader(open(self.env.house.metaDataFile, 'r'))
        headers = next(csvFile)

        for row in csvFile:
            modelCategoryMapping[row[headers.index('model_id')]] = {
                headers[x]: row[x]
                for x in range(2, len(headers))  # 0 is index, 1 is model_id
            }

        for i in levelsToExplore:
            for j in range(len(data['levels'][i]['nodes'])):
                assert data['levels'][i]['nodes'][j]['type'] != 'Box'

                if 'valid' in data['levels'][i]['nodes'][j]:
                    assert data['levels'][i]['nodes'][j]['valid'] == 1

                # Rooms
                if data['levels'][i]['nodes'][j]['type'] == 'Room':
                    if 'roomTypes' not in data['levels'][i]['nodes'][j]:
                        continue

                    # Can rooms have more than one type?
                    # Yes, they can; just found ['Living_Room', 'Dining_Room', 'Kitchen']
                    # assert len(data['levels'][i]['nodes'][j]['roomTypes']) <= 3

                    roomType = [
                        # ' '.join(x.lower().split('_'))
                        x.lower()
                        for x in data['levels'][i]['nodes'][j]['roomTypes']
                    ]

                    nodes = data['levels'][i]['nodes'][j][
                        'nodeIndices'] if 'nodeIndices' in data['levels'][i][
                            'nodes'][j] else []
                    rooms.append({
                        'type':
                        roomType,
                        'bbox':
                        data['levels'][i]['nodes'][j]['bbox'],
                        'nodes':
                        nodes,
                        'model_id':
                        data['levels'][i]['nodes'][j]['modelId']
                    })

                # Objects
                elif data['levels'][i]['nodes'][j]['type'] == 'Object':
                    if 'materials' not in data['levels'][i]['nodes'][j]:
                        material = []
                    else:
                        material = data['levels'][i]['nodes'][j]['materials']
                    objects[data['levels'][i]['nodes'][j]['id']] = {
                        'id':
                        data['levels'][i]['nodes'][j]['id'],
                        'model_id':
                        data['levels'][i]['nodes'][j]['modelId'],
                        'fine_class':
                        modelCategoryMapping[data['levels'][i]['nodes'][j][
                            'modelId']]['fine_grained_class'],
                        'coarse_class':
                        modelCategoryMapping[data['levels'][i]['nodes'][j][
                            'modelId']]['coarse_grained_class'],
                        'bbox':
                        data['levels'][i]['nodes'][j]['bbox'],
                        'mat':
                        material
                    }

        return rooms, objects

    # Spawn at a randomly selected point in a particular room
    def spawn_room(self, room=None):
        if room == None:
            return False, None

        target_room = '_'.join(room.lower().split(' '))

        if self.env.house.hasRoomType(target_room) == False:
            return False, None

        rooms = self.env.house._getRooms(target_room)
        room = np.random.choice(rooms)

        gx1, gy1, gx2, gy2 = self.env.house._getRoomBounds(room)

        available_coords = []
        for x in range(gx1, gx2 + 1):
            for y in range(gy1, gy2 + 1):
                if self.env.house.moveMap[x, y] > 0:
                    available_coords.append((x, y))

        # print(available_coords)
        spawn_coord_idx = np.random.choice(len(available_coords))
        spawn_coord = available_coords[spawn_coord_idx]

        return spawn_coord, room

    # Spawn close to an object
    # If room given, look for object within room
    def spawn_object(self, obj=None, room=None):
        if object == None:
            return False, None

        if isinstance(obj, list) == False:
            obj = [obj]

        is_door = False
        if 'door' in obj:
            is_door = True

        target_obj = ['_'.join(x.lower().split(' ')) for x in obj]

        if room != None:
            if 'nodeIndices' in room:
                objs = [
                    self.objects['0_' + str(x)] for x in room['nodeIndices']
                    if self.objects['0_' + str(x)]['fine_class'] in target_obj
                ]
            else:
                objs = [
                    self.objects['0_' + str(x)] for x in room['nodes']
                    if self.objects['0_' + str(x)]['fine_class'] in target_obj
                ]
        else:
            obj_id_list = list(
                itertools.chain.from_iterable(
                    [x['nodes'] for x in self.rooms if x['type'] != []]))
            objs = [
                self.objects['0_' + str(x)] for x in obj_id_list
                if self.objects['0_' + str(x)]['fine_class'] in target_obj
            ]

        if len(objs) == 0:
            return False, None, None

        obj_idx = np.random.choice(len(objs))
        obj = objs[obj_idx]

        self.target_obj_class = obj['fine_class'].lower()

        gx1, gy1, gx2, gy2 = self.env.house._getRoomBounds(obj)

        if room == None:
            obj_node_idx = int(obj['id'][2:])
            room = [
                x for x in self.env.house.all_rooms
                if 'nodeIndices' in x and obj_node_idx in x['nodeIndices']
            ][0]

        self.set_target_object(obj, room)

        available_x, available_y = np.where(self.env.house.connMap == 0)

        if len(available_x) == 0:
            return False, None, None

        spawn_coords = []
        for i in range(len(available_x)):
            spawn_coords.append((available_x[i], available_y[i]))

        return spawn_coords, obj, room

    # analogous to `setTargetRoom` in the House3D API
    def set_target_object(self, obj, room):
        object_tp = room['id'] + '_' + obj['id'] + '_' + obj['fine_class'].lower(
        )
        # Caching
        if object_tp in self.env.house.connMapDict:
            self.env.house.connMap, self.env.house.connectedCoors, self.env.house.inroomDist, self.env.house.maxConnDist = self.env.house.connMapDict[
                object_tp]
            return True  # object changed!
        elif os.path.exists(
                os.path.join(
                    self.target_obj_conn_map_dir,
                    self.env.house.house['id'] + '_' + object_tp + '.npy')):
            self.env.house.connMap = np.load(
                os.path.join(
                    self.target_obj_conn_map_dir,
                    self.env.house.house['id'] + '_' + object_tp + '.npy'))

            if self.env.house.connMap.shape[0] == self.env.house.n_row+1:
                self.env.house.connectedCoors, self.env.house.inroomDist, self.env.house.maxConnDist = None, None, None
                return True

        self.env.house.connMap = connMap = np.ones(
            (self.env.house.n_row + 1, self.env.house.n_row + 1),
            dtype=np.int32) * -1
        self.env.house.inroomDist = inroomDist = np.ones(
            (self.env.house.n_row + 1, self.env.house.n_row + 1),
            dtype=np.float32) * -1
        dirs = [[0, 1], [1, 0], [-1, 0], [0, -1]]
        que = []
        flag_find_open_components = True

        _ox1, _, _oy1 = obj['bbox']['min']
        _ox2, _, _oy2 = obj['bbox']['max']
        ocx, ocy = (_ox1 + _ox2) / 2, (_oy1 + _oy2) / 2
        ox1, oy1, ox2, oy2 = self.env.house.rescale(_ox1, _oy1, _ox2, _oy2)

        for _ in range(2):
            _x1, _, _y1 = room['bbox']['min']
            _x2, _, _y2 = room['bbox']['max']
            cx, cy = (_x1 + _x2) / 2, (_y1 + _y2) / 2
            x1, y1, x2, y2 = self.env.house.rescale(_x1, _y1, _x2, _y2)

            curr_components = self.env.house._find_components(
                x1,
                y1,
                x2,
                y2,
                dirs=dirs,
                return_open=flag_find_open_components
            )  # find all the open components
            if len(curr_components) == 0:
                print('No space found! =(')
                raise ValueError('no space')
            if isinstance(curr_components[0],
                          list):  # join all the coors in the open components
                curr_major_coors = list(itertools.chain(*curr_components))
            else:
                curr_major_coors = curr_components
            min_dist_to_center, min_dist_to_edge = 1e50, 1e50
            for x, y in curr_major_coors:
                ###
                # Compute minimum dist to edge here
                if x in range(ox1, ox2):
                    dx = 0
                elif x < ox1:
                    dx = ox1 - x
                else:
                    dx = x - ox2

                if y in range(oy1, oy2):
                    dy = 0
                elif y < oy1:
                    dy = oy1 - y
                else:
                    dy = y - oy2

                assert dx >= 0 and dy >= 0

                if dx != 0 or dy != 0:
                    dd = np.sqrt(dx**2 + dy**2)
                elif dx == 0:
                    dd = dy
                else:
                    dd = dx

                if dd < min_dist_to_edge:
                    min_dist_to_edge = int(np.ceil(dd))
                ###
                tx, ty = self.env.house.to_coor(x, y)
                tdist = np.sqrt((tx - ocx)**2 + (ty - ocy)**2)
                if tdist < min_dist_to_center:
                    min_dist_to_center = tdist
                inroomDist[x, y] = tdist
            margin = min_dist_to_edge + 1
            for x, y in curr_major_coors:
                inroomDist[x, y] -= min_dist_to_center
            for x, y in curr_major_coors:
                if x in range(ox1 - margin, ox2 + margin) and y in range(
                        oy1 - margin, oy2 + margin):
                    connMap[x, y] = 0
                    que.append((x, y))
            if len(que) > 0: break
            if flag_find_open_components:
                flag_find_open_components = False
            else:
                break
            raise ValueError

        ptr = 0
        self.env.house.maxConnDist = 1
        while ptr < len(que):
            x, y = que[ptr]
            cur_dist = connMap[x, y]
            ptr += 1
            for dx, dy in dirs:
                tx, ty = x + dx, y + dy
                if self.env.house.inside(tx, ty) and self.env.house.canMove(
                        tx, ty) and not self.env.house.isConnect(tx, ty):
                    que.append((tx, ty))
                    connMap[tx, ty] = cur_dist + 1
                    if cur_dist + 1 > self.env.house.maxConnDist:
                        self.env.house.maxConnDist = cur_dist + 1
        self.env.house.connMapDict[object_tp] = (connMap, que, inroomDist,
                                                 self.env.house.maxConnDist)
        np.save(
            os.path.join(
                self.target_obj_conn_map_dir,
                self.env.house.house['id'] + '_' + object_tp + '.npy'),
            connMap)
        self.connectedCoors = que
        print(' >>>> ConnMap Cached!')
        return True  # room changed!

    def _load_semantic_classes(self, color_file=None):
        if color_file == None:
            color_file = self.env.config['colorFile']

        self.semantic_classes = {}

        with open(color_file) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                c = np.array((row['r'], row['g'], row['b']), dtype=np.uint8)
                fine_cat = row['name'].lower()
                self.semantic_classes[fine_cat] = c

        return self.semantic_classes

    def _get_best_yaw_obj_from_pos(self, obj_id, grid_pos, height=1.0):
        obj = self.objects[obj_id]
        obj_fine_class = obj['fine_class']

        cx, cy = self.env.house.to_coor(grid_pos[0], grid_pos[1])

        self.env.cam.pos.x = cx
        self.env.cam.pos.y = height
        self.env.cam.pos.z = cy

        best_yaw, best_coverage = None, 0

        for yaw in self.angles:
            self.env.cam.yaw = yaw
            self.env.cam.updateDirection()

            seg = self.env.render(mode='semantic')
            c = self.semantic_classes[obj_fine_class.lower()]
            mask = np.all(seg == c, axis=2)
            coverage = np.sum(mask) / (seg.shape[0] * seg.shape[1])

            if best_yaw == None:
                best_yaw = yaw
                best_coverage = coverage
            else:
                if coverage > best_coverage:
                    best_yaw = yaw
                    best_coverage = coverage

        return best_yaw, best_coverage

    def _get_best_view_obj(self,
                           obj,
                           coverage_thres=0.5,
                           dist_add=0.5,
                           robot_height=False):
        bbox = obj['bbox']
        obj_fine_class = obj['fine_class']

        obj_max = np.asarray(bbox['max'])
        obj_min = np.asarray(bbox['min'])
        obj_center = (obj_min + obj_max) / 2

        c_x, c_y, c_z = obj_center
        max_radius = np.sqrt(
            (obj_max[0] - obj_min[0]) * (obj_max[0] - obj_min[0]) +
            (obj_max[2] - obj_min[2]) * (obj_max[2] - obj_min[2])) / 2.0
        max_radius += dist_add

        best_pos = None
        best_coverage = 0

        returned_pos_cov = []

        for yaw in self.angles:
            pos = [
                c_x - max_radius * np.cos(yaw * (2 * np.pi) / 360.0), c_y,
                c_z - max_radius * np.sin(yaw * (2 * np.pi) / 360.0), yaw
            ]

            if robot_height == True:
                pos[1] = min(max(0.75, c_y), 2.00)

            self.env.cam.pos.x = pos[0]
            self.env.cam.pos.y = pos[1]
            self.env.cam.pos.z = pos[2]
            self.env.cam.yaw = pos[3]

            self.env.cam.updateDirection()

            seg = self.env.render(mode='semantic')
            c = self.semantic_classes[obj_fine_class.lower()]
            mask = np.all(seg == c, axis=2)
            coverage = np.sum(mask) / (seg.shape[0] * seg.shape[1])

            returned_pos_cov.append([pos, coverage])

            if coverage > coverage_thres:
                return pos, coverage, returned_pos_cov
            elif coverage > best_coverage:
                best_coverage = coverage
                best_pos = pos

        return best_pos, best_coverage, returned_pos_cov