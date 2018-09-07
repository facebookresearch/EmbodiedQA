import argparse
from datetime import datetime
from heapq import heappush, heappop
import json
import logging
import numpy as np
import os
import pickle
import random
import sys
import traceback
from time import time
from tqdm import tqdm

from House3D import objrender, Environment, load_config

sys.path.insert(0, '../../utils/')
from house3d import House3DUtils


def heuristic_estimate(agent_continous, target_continous):
    return np.sqrt((agent_continous[0] - target_continous[0]) ** 2 + (agent_continous[1] - target_continous[1]) ** 2)


def add_neighbors(h3d, points_queue, point, distances_source, prev_pos, target_continous):
    dist = distances_source[point[:3]]

    # check and add edges for angle movements
    for angle in [h3d.rotation_sensitivity, -h3d.rotation_sensitivity]:
        h3d.env.reset(x=point[3], y=point[4], yaw=point[2])
        h3d.env.rotate(angle)
        assert h3d.env.cam.yaw - point[2] == angle

        cand_point = (point[0], point[1], (point[2] + 180 + angle) % 360 - 180)
        if cand_point not in distances_source:
            distances_source[cand_point] = dist + 1
            prev_pos[cand_point] = point
            f_dist = dist + 1 + heuristic_estimate((point[3], point[4]), target_continous)
            cand_point = (cand_point[0], cand_point[1], cand_point[2],
                          h3d.env.cam.pos.x, h3d.env.cam.pos.z)
            heappush(points_queue, (f_dist, cand_point))

    # TODO(akadian): optimization -> check if node has been visited previously before stepping through the environment

    # check and add edge for forward movement
    h3d.env.reset(x=point[3], y=point[4], yaw=point[2])
    pre_pos = [h3d.env.cam.pos.x, h3d.env.cam.pos.z, h3d.env.cam.yaw]
    h3d.env.move_forward(dist_fwd=h3d.move_sensitivity, dist_hor=0)
    post_pos = [h3d.env.cam.pos.x, h3d.env.cam.pos.z, h3d.env.cam.yaw]
    if all([np.abs(pre_pos[x] - post_pos[x]) < 1e-9 for x in range(3)]):
        return

    tx, ty = h3d.env.house.to_grid(h3d.env.cam.pos.x, h3d.env.cam.pos.z)
    cand_point = (tx, ty, h3d.env.cam.yaw)
    if cand_point not in distances_source:
        distances_source[cand_point] = dist + 1
        prev_pos[cand_point] = point
        f_dist = dist + 1 + heuristic_estimate((h3d.env.cam.pos.x, h3d.env.cam.pos.z), target_continous)
        cand_point = (cand_point[0], cand_point[1], cand_point[2],
                      h3d.env.cam.pos.x, h3d.env.cam.pos.z)
        heappush(points_queue, (f_dist, cand_point))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-questions_json',
                        default='/private/home/akadian/eqa-data/suncg-data/eqa_v1.json')
    parser.add_argument('-graph_dir', default='/private/home/akadian/eqa-data/suncg-data/a-star')
    parser.add_argument('-target_obj_conn_map_dir',
                        default='/private/home/akadian/eqa-data/suncg-data/a-star/target_obj_conn_map_dir')
    parser.add_argument('-shortest_path_dir')
    parser.add_argument('-invalids_dir', default='/private/home/akadian/eqa-data/suncg-data/invalids/')
    parser.add_argument('-env_id', default=None)
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-map_resolution', default=1000, type=int)
    parser.add_argument('-seed', type=int, required=True)
    parser.add_argument('-check_validity', action="store_true")
    parser.add_argument('-log_path', default=None)
    parser.add_argument('-source_candidate_fraction', type=float, default=0.05)
    args = parser.parse_args()

    if args.log_path is None:
        args.log_path = 'seed_{}_resolution_{}.{}.log'.format(args.seed, args.map_resolution,
                                                              str(datetime.now()).replace(' ', '_'))
    logging.basicConfig(filename=args.log_path, level=logging.INFO,
                        format='%(asctime)-15s %(message)s')
    random.seed(args.seed)
    np.random.seed(args.seed)
    if not os.path.exists(args.shortest_path_dir):
        os.makedirs(args.shortest_path_dir)
    args.gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    args.gpus = [int(x) for x in args.gpus]
    # create specific directories corresponding to the resolution
    args.graph_dir = os.path.join(args.graph_dir, str(args.map_resolution))
    args.target_obj_conn_map_dir = os.path.join(args.target_obj_conn_map_dir, str(args.map_resolution))
    # load house3d renderer
    cfg = load_config('../../House3D/tests/config.json')
    api_thread = objrender.RenderAPIThread(w=224, h=224, device=args.gpus[0])
    # load envs list from questions json
    data = json.load(open(args.questions_json, 'r'))
    qns = data['questions']
    if args.env_id is None:
        envs = sorted(list(set(qns.keys())))
    else:
        envs = [args.env_id]
    random.shuffle(envs)
    invalid = []

    count_path_found = 0
    count_valid = 0
    count_path_not_found = 0
    count_no_source_cands = 0
    shortest_path_lengths = []

    for h in tqdm(envs):
        # `scn2scn` from suncg-toolbox segfaults for this env :/
        if h == '436d655f24d385512e1e782b5ba88c6b':
            continue
        for q in qns[h]:
            logging.info("count_path_found: {}".format(count_path_found))
            logging.info("count_valid: {}".format(count_valid))
            logging.info("count_path_not_found: {}".format(count_path_not_found))
            logging.info("count_no_source_cands: {}".format(count_no_source_cands))
            if len(shortest_path_lengths) > 0:
                logging.info("shortest path length mean: {}, median: {}, min: {}, max: {}".format(
                    np.mean(shortest_path_lengths), np.median(shortest_path_lengths),
                    np.min(shortest_path_lengths), np.max(shortest_path_lengths)))
            logging.info("env, question pair: {}_{}".format(h, q['id']))
            logging.info("{} {} {}".format(h, q['question'], q['answer']))
            env = Environment(api_thread, h, cfg, ColideRes=args.map_resolution)
            h3d = House3DUtils(env, graph_dir=args.graph_dir,
                               target_obj_conn_map_dir=args.target_obj_conn_map_dir,
                               build_graph=False)

            if os.path.exists(os.path.join(args.shortest_path_dir, "{}_{}.pkl".format(h, q['id']))):
                logging.info("Shortest path exists")
                continue

            # set target object
            bbox_obj = [x for x in q['bbox'] if x['type'] == 'object' and x['target'] is True][0]
            obj_id = []
            for x in h3d.objects:
                if all([h3d.objects[x]['bbox']['min'][i] == bbox_obj['box']['min'][i] for i in range(3)]) and \
                        all([h3d.objects[x]['bbox']['max'][i] == bbox_obj['box']['max'][i] for i in range(3)]):
                    obj_id.append(x)
                    if h3d.objects[x]['fine_class'] != bbox_obj['name']:
                        logging.info('Name not matched {} {}'.format(h3d.objects[x]['fine_class'], bbox_obj['name']))
            assert len(obj_id) == 1
            bbox_room = [x for x in q['bbox'] if x['type'] == 'room' and x['target'] is False][0]
            target_room = False
            for room in h3d.env.house.all_rooms:
                if all([room['bbox']['min'][i] == bbox_room['box']['min'][i] for i in range(3)]) and \
                        all([room['bbox']['max'][i] == bbox_room['box']['max'][i] for i in range(3)]):
                    target_room = room
                    break
            target_obj = obj_id[0]
            h3d.set_target_object(h3d.objects[target_obj], target_room)

            # sample a close enough target point
            target_point_cands = np.argwhere((env.house.connMap >= 0) & (env.house.connMap <= 5))
            target_point_idx = np.random.choice(target_point_cands.shape[0])
            target_yaw, best_coverage = h3d._get_best_yaw_obj_from_pos(
                target_obj,
                [target_point_cands[target_point_idx][0],
                 target_point_cands[target_point_idx][1]],
                height=1.0)
            target_point = (target_point_cands[target_point_idx][0],
                            target_point_cands[target_point_idx][1],
                            target_yaw)

            # graph creation used for selecting a source point
            t1 = time()
            if os.path.exists(os.path.join(h3d.graph_dir, h3d.env.house.house['id'] + '_' + target_obj + '.pkl')):
                print('loading graph')
                h3d.load_graph(os.path.join(h3d.graph_dir, h3d.env.house.house['id'] + '_' + target_obj + '.pkl'))
            else:
                print('building graph')
                h3d.build_graph(
                    save_path=os.path.join(h3d.graph_dir, h3d.env.house.house['id'] + '_' + target_obj + '.pkl'))

            connmap_values = env.house.connMap.flatten()
            connmap_values.sort()
            # threshold for --source_candidate_fraction number of points
            thresh = connmap_values[int((1.0 - args.source_candidate_fraction) * connmap_values.shape[0])]
            source_point_cands = np.argwhere((env.house.connMap != -1) & (env.house.connMap >= thresh))
            if thresh < 50:
                # sanity check to prevent scenario when agent is spawned close to target location
                logging.info("No source candidates")
                invalid.append(h)
                count_no_source_cands += 1
                continue
            t2 = time()
            logging.info("Time spent for graph creation {:.6f}s".format(t2 - t1))

            for it in range(10):
                logging.info("Try: {}".format(it))
                try:
                    source_point_idx = np.random.choice(source_point_cands.shape[0])
                    source_point = (source_point_cands[source_point_idx][0], source_point_cands[source_point_idx][1],
                                    np.random.choice(h3d.angles))

                    # A* for shortest path
                    t3 = time()
                    target_x, target_y, target_yaw = target_point
                    source_continous = h3d.env.house.to_coor(source_point[0], source_point[1], shft=True)
                    target_continous = h3d.env.house.to_coor(target_x, target_y, shft=True)
                    points_queue = []
                    distances_source = dict()
                    prev_pos = dict()
                    distances_source[source_point] = 0
                    prev_pos[source_point] = (-1.0, -1.0, -1.0, -1.0, -1.0)

                    # schema for point in points_queue:
                    # (x-grid-location, y-grid-location, yaw, x-continous-coordinate, y-continous-coordinate)
                    source_point = (source_point[0], source_point[1], source_point[2],
                                    source_continous[0], source_continous[1])
                    heappush(points_queue, (heuristic_estimate(source_continous, target_continous), source_point))

                    while True:
                        if len(points_queue) == 0:
                            count_path_not_found += 1
                            logging.info("A* not able to find path to target")
                            raise ValueError("Path not found to target {} {}".format(source_point[:3], target_point))
                        f_dist, point = heappop(points_queue)
                        add_neighbors(h3d, points_queue, point, distances_source, prev_pos, target_continous)
                        if point[0] == target_x and point[1] == target_y and point[2] == target_yaw:
                            # store path
                            shortest_path_nodes = []
                            while True:
                                shortest_path_nodes.append(point)
                                point = prev_pos[point[:3]]
                                if point[0] == -1:
                                    break
                            shortest_path_nodes.reverse()
                            break
                    t4 = time()
                    logging.info("Time spent for coupled graph generation and A*: {:.6f}s".format(t4 - t3))

                    # bookkeeping
                    act_q, pos_q, coord_q, actual_q = [], [], [], []
                    episode_images = []
                    movemap = None
                    for i in range(len(shortest_path_nodes) - 1):
                        u = shortest_path_nodes[i]
                        v = shortest_path_nodes[i + 1]
                        pos_q.append((float(u[3]), 1.0, float(u[4]), float(u[2])))
                        coord_q.append(h3d.env.house.to_grid(u[3], u[4]))
                        curr_x, curr_y, curr_yaw = u[3], u[4], u[2]
                        next_x, next_y, next_yaw = v[3], v[4], v[2]
                        if curr_yaw != next_yaw:
                            if next_yaw == 171 and curr_yaw == -180:
                                act_q.append(1)
                            elif next_yaw == -180 and curr_yaw == 171:
                                act_q.append(2)
                            elif next_yaw < curr_yaw:
                                act_q.append(1)
                            else:
                                act_q.append(2)
                        else:
                            act_q.append(0)
                    pos_q.append((shortest_path_nodes[-1][3], 1.0,
                                  shortest_path_nodes[-1][4], shortest_path_nodes[-1][2]))
                    act_q.append(3)

                    if args.check_validity:
                        h3d.env.reset(x=pos_q[0][0], y=pos_q[0][2], yaw=pos_q[0][3])
                        h3d_yaw = pos_q[0][3]  # dummy yaw limited to [-180, 180)
                        actual_q.append((float(h3d.env.cam.pos.x), 1.0, float(h3d.env.cam.pos.z),
                                         float(h3d.env.cam.yaw)))
                        for i, action in enumerate(act_q[:-1]):
                            pre_pos = [h3d.env.cam.pos.x, h3d.env.cam.pos.z, h3d.env.cam.yaw]
                            img, _, episode_done = h3d.step(action)
                            episode_images.append(img)
                            post_pos = [h3d.env.cam.pos.x, h3d.env.cam.pos.z, h3d.env.cam.yaw]
                            actual_q.append((float(h3d.env.cam.pos.x), 1.0, float(h3d.env.cam.pos.z),
                                             float(h3d.env.cam.yaw)))
                            if all([np.abs(pre_pos[x] - post_pos[x]) < 1e-9 for x in range(3)]):
                                raise ValueError("Invalid action")
                            angle_delta = post_pos[2] - pre_pos[2]
                            h3d_yaw = (h3d_yaw + 180 + angle_delta) % 360 - 180
                            assert np.abs(h3d.env.cam.pos.x - pos_q[i + 1][0]) < 1e-3
                            assert np.abs(h3d.env.cam.pos.z - pos_q[i + 1][2]) < 1e-3
                            assert h3d_yaw == pos_q[i + 1][3]
                        count_valid += 1
                        movemap = h3d.env.house._showMoveMap(visualize=False)
                        logging.info("Valid")

                    result = {
                        "actions": act_q,
                        "actual_q": actual_q,
                        "answer": q['answer'],
                        "coordinates": coord_q,
                        "images": episode_images,
                        "movemap": movemap,
                        "positions": pos_q,
                        "question": q['question'],
                    }
                    with open(os.path.join(args.shortest_path_dir, "{}_{}.pkl".format(h, q['id'])), "wb") as f:
                        pickle.dump(result, f)
                        logging.info("Saved {}_{}.pkl".format(h, q['id']))
                        logging.info("Length of shortest path: {}".format(len(shortest_path_nodes)))
                        shortest_path_lengths.append(len(shortest_path_nodes))
                    count_path_found += 1
                    break
                except KeyboardInterrupt:
                    raise
                except:
                    invalid.append("env, question pair: {}_{}".format(h, q['id']))
                    traceback.print_exc()


if __name__ == "__main__":
    main()

