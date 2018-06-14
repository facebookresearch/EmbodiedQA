# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import csv
import random
import argparse
import operator
import numpy as np
import os, sys, json
from tqdm import tqdm
from scipy import spatial
from numpy.random import choice
from random import shuffle

from house_parse import HouseParse
from question_string_builder import QuestionStringBuilder

from nltk.stem import WordNetLemmatizer

random.seed(0)
np.random.seed(0)

class roomEntity():
    translations = {
        'toilet': 'bathroom',
        'guest room': 'bedroom',
        'child room': 'bedroom',
    }

    def __init__(self, name, bbox, meta):
        self.name = list(
            set([
                self.translations[str(x)]
                if str(x) in self.translations else str(x) for x in name
            ]))
        self.bbox = bbox
        self.meta = meta
        self.type = 'room'

        self.name.sort(key=str.lower)
        self.entities = self.objects = []

    def addObject(self, object_ent):
        self.objects.append(object_ent)

    def isValid(self):
        return len(self.objects) != 0


class objectEntity():
    translations = {
        'bread': 'food',
        'hanging_kitchen_cabinet': 'kitchen_cabinet',
        'teapot': 'kettle',
        'coffee_kettle': 'kettle',
        'range_hood_with_cabinet': 'range_hood',
        'dining_table': 'table',
        'coffee_table': 'table',
        'game_table': 'table',
        'office_chair': 'chair',
        'bench_chair': 'chair',
        'chair_set': 'chair',
        'armchair': 'chair',
        'fishbowl': 'fish_tank/bowl',
        'fish_tank': 'fish_tank/bowl',
        'single_bed': 'bed',
        'double_bed': 'bed',
        'baby_bed': 'bed'
    }

    def __init__(self, name, bbox, meta, obj_id=False):
        if name in self.translations: self.name = self.translations[name]
        else: self.name = name
        self.bbox = bbox
        self.meta = meta
        self.type = 'object'
        self.id = obj_id

        self.entities = self.rooms = []

    def addRoom(self, room_ent):
        self.rooms.append(room_ent)

    def isValid(self):
        return len(self.rooms) != 0


class Engine():
    '''
        Templates and functional forms.
    '''

    template_defs = {
        'location': [
            'filter.objects', 'unique.objects', 'blacklist.location',
            'query.room'
        ],
        'count': [
            'filter.rooms', 'unique.rooms', 'filter.objects',
            'blacklist.count', 'query.count'
        ],
        'room_count': ['filter.rooms', 'query.room_count'],
        'global_object_count':
        ['filter.objects', 'blacklist.count', 'query.global_object_count'],
        'room_object_count':
        ['filter.objects', 'blacklist.exist', 'query.room_object_count'],
        'exist': [
            'filter.rooms', 'unique.rooms', 'filter.objects',
            'blacklist.exist', 'query.exist'
        ],
        'exist_logical': [
            'filter.rooms', 'unique.rooms', 'filter.objects',
            'blacklist.exist', 'query.logical'
        ],
        'color':
        ['filter.objects', 'unique.objects', 'blacklist.color', 'query.color'],
        'color_room': [
            'filter.rooms', 'unique.rooms', 'filter.objects', 'unique.objects',
            'blacklist.color_room', 'query.color_room'
        ],
        'relate': [
            'filter.objects', 'unique.objects', 'blacklist.relate', 'relate',
            'query.object'
        ],
        'relate_room': [
            'filter.rooms', 'unique.rooms', 'filter.objects', 'unique.objects',
            'blacklist.relate', 'relate', 'query.object_room'
        ],
        'dist_compare': [
            'filter.rooms', 'unique.rooms', 'filter.objects', 'unique.objects',
            'blacklist.dist_compare', 'distance', 'query.compare'
        ]
    }

    templates = {
        'location':
        'what room <AUX> the <OBJ> located in?',
        'count':
        'how many <OBJ-plural> are in the <ROOM>?',
        'room_count':
        'how many <ROOM-plural> are in the house?',
        'room_object_count':
        'how many rooms in the house have <OBJ-plural> in them?',
        'global_object_count':
        'how many <OBJ-plural> are there in all <ROOM-plural> across the house?',
        'exist':
        '<AUX> there <ARTICLE> <OBJ> in the <ROOM>?',
        'exist_logic':
        '<AUX> there <ARTICLE> <OBJ1> <LOGIC> <ARTICLE> <OBJ2> in the <ROOM>?',
        'color':
        'what color <AUX> the <OBJ>?',
        'color_room':
        'what color <AUX> the <OBJ> in the <ROOM>?',
        # prepositions of place
        'above':
        'what is above the <OBJ>?',
        'on':
        'what is on the <OBJ>?',
        'below':
        'what is below the <OBJ>?',
        'under':
        'what is under the <OBJ>?',
        'next_to':
        'what is next to the <OBJ>?',
        'above_room':
        'what is above the <OBJ> in the <ROOM>?',
        'on_room':
        'what is on the <OBJ> in the <ROOM>?',
        'below_room':
        'what is below the <OBJ> in the <ROOM>?',
        'under_room':
        'what is under the <OBJ> in the <ROOM>?',
        'next_to_room':
        'what is next to the <OBJ> in the <ROOM>?',
        # object distance comparisons
        'closer_room':
        'is the <OBJ> closer to the <OBJ> than to the <OBJ> in the <ROOM>?',
        'farther_room':
        'is the <OBJ> farther from the <OBJ> than from the <OBJ> in the <ROOM>?'
    }

    blacklist_objects = {
        'location': [
            'column', 'door', 'kitchen_cabinet', 'kitchen_set',
            'hanging_kitchen_cabinet', 'switch', 'range_hood_with_cabinet',
            'game_table', 'headstone', 'pillow', 'range_oven_with_hood',
            'glass', 'roof', 'cart', 'window', 'headphones_on_stand', 'coffin',
            'book', 'toy', 'workplace', 'range_hood', 'trinket', 'ceiling_fan',
            'beer', 'books', 'magazines', 'shelving', 'partition',
            'containers', 'container', 'grill', 'stationary_container',
            'bottle', 'outdoor_seating', 'stand', 'place_setting', 'arch',
            'household_appliance', 'pet', 'person', 'chandelier', 'decoration'
        ],
        'count': [
            'container', 'containers', 'stationary_container', 'switch',
            'place_setting', 'workplace', 'grill', 'shelving', 'person', 'pet',
            'chandelier', 'household_appliance', 'decoration', 'trinket',
            'kitchen_set', 'headstone', 'arch', 'ceiling_fan', 'glass', 'roof',
            'outdoor_seating', 'stand', 'kitchen_cabinet', 'coffin', 'beer',
            'book', 'books'
        ],
        'exist': [
            'container', 'containers', 'stationary_container', 'decoration',
            'trinket', 'place_setting', 'workplace', 'grill', 'switch',
            'window', 'door', 'column', 'person', 'pet', 'chandelier',
            'household_appliance', 'ceiling_fan', 'arch', 'book', 'books',
            'glass', 'roof', 'shelving', 'outdoor_seating', 'stand',
            'kitchen_cabinet', 'kitchen_set', 'coffin', 'headstone', 'beer'
        ],
        'color': [
            'container', 'containers', 'stationary_container', 'candle',
            'coffee_table', 'column', 'door', 'floor_lamp', 'mirror', 'person',
            'rug', 'sofa', 'stairs', 'outdoor_seating', 'kitchen_cabinet',
            'kitchen_set', 'switch', 'storage_bench', 'table_lamp', 'vase',
            'candle', 'roof', 'stand', 'beer', 'chair', 'chandelier',
            'coffee_table', 'column', 'trinket', 'grill', 'book', 'books',
            'curtain', 'desk', 'door', 'floor_lamp', 'hanger', 'workplace',
            'glass', 'headstone', 'kitchen_set', 'mirror', 'plant', 'shelving',
            'place_setting', 'ceiling_fan', 'stairs', 'storage_bench',
            'switch', 'table_lamp', 'vase', 'decoration', 'coffin',
            'wardrobe_cabinet', 'window', 'pet', 'cup', 'arch',
            'household_appliance'
        ],
        'color_room': [
            'column', 'door', 'kitchen_cabinet', 'kitchen_set', 'mirror',
            'household_appliance', 'decoration', 'place_setting', 'book',
            'person', 'stairs', 'switch', 'pet', 'chandelier', 'container',
            'containers', 'stationary_container', 'trinket', 'coffin', 'books',
            'ceiling_fan', 'workplace', 'glass', 'grill', 'roof', 'shelving',
            'outdoor_seating', 'stand', 'headstone', 'arch', 'beer'
        ],
        'relate': [
            'office_chair', 'column', 'door', 'switch', 'partition',
            'household_appliance', 'decoration', 'place_setting', 'book',
            'person', 'pet', 'chandelier', 'container', 'containers',
            'stationary_container', 'trinket', 'stand', 'kitchen_set', 'arch',
            'books', 'ceiling_fan', 'workplace', 'glass', 'grill', 'roof',
            'shelving', 'outdoor_seating', 'kitchen_cabinet', 'coffin',
            'headstone', 'beer'
        ],
        'dist_compare': [
            'column', 'door', 'switch', 'person', 'household_appliance',
            'decoration', 'trinket', 'place_setting', 'coffin', 'book'
            'cup', 'chandelier', 'arch', 'pet', 'container', 'containers',
            'stationary_container', 'shelving', 'stand', 'kitchen_set',
            'books', 'ceiling_fan', 'workplace', 'glass', 'grill', 'roof',
            'outdoor_seating', 'kitchen_cabinet', 'headstone', 'beer'
        ]
    }

    blacklist_rooms = [
        'loggia', 'storage', 'guest room', 'hallway', 'wardrobe', 'hall',
        'boiler room', 'terrace', 'room', 'entryway', 'aeration', 'lobby',
        'office', 'freight elevator', 'passenger elevator'
    ]

    use_threshold_size = True
    use_blacklist = True

    def __init__(
            self,
            debug=False,
            object_counts_by_room_file="data/obj_counts_by_room.json"
    ):
        self.template_fns = {
            'filter': self.filter,
            'unique': self.unique,
            'query': self.query,
            'relate': self.relate,
            'distance': self.distance,
            'blacklist': self.blacklist,
            'thresholdSize': self.thresholdSize
        }

        self.query_fns = {
            'query_room': self.queryRoom,
            'query_count': self.queryCount,
            'query_room_count': self.queryRoomCounts,
            'query_global_object_count': self.queryGlobalObjectCounts,
            'query_room_object_count': self.queryRoomObjectCounts,
            'query_exist': self.queryExist,
            'query_logical': self.queryLogical,
            'query_color': self.queryColor,
            'query_color_room': self.queryColorRoom,
            'query_object': self.queryObject,
            'query_object_room': self.queryObjectRoom,
            'query_compare': self.queryCompare
        }

        self.debug = debug
        self.ent_queue = None
        self.q_str_builder = QuestionStringBuilder()
        self.q_obj_builder = self.questionObjectBuilder

        # update
        if os.path.isfile(object_counts_by_room_file) == True:
            self.global_obj_by_room = json.load(
                open(object_counts_by_room_file, 'r'))
            self.negative_exists = {}
        else:
            print('Not loading data/obj_counts_by_room.json')

        # load colors
        self.env_obj_color_map = json.load(open('data/obj_colors.json', 'r'))

    def cacheHouse(self, Hp):
        self.house = Hp

        self.entities = {'rooms': [], 'objects': []}

        for i in self.house.rooms:
            room = roomEntity(i['type'], i['bbox'], i)
            for j in room.meta['nodes']:
                obj = objectEntity(
                    self.house.objects['0_' + str(j)]['fine_class'],
                    self.house.objects['0_' + str(j)]['bbox'],
                    self.house.objects['0_' + str(j)],
                    obj_id='0_' + str(j))
                room.addObject(obj)
                obj.addRoom(room)

                self.entities['objects'].append(obj)

            self.entities['rooms'].append(room)

        self.isValid()

    def isValid(self):
        # print('checking validity...')
        for i in self.entities['rooms']:
            if i.isValid() == False and self.debug == True:
                print('ERROR', i.meta)
                continue

        for i in self.entities['objects']:
            if i.isValid() == False and self.debug == True:
                print('ERROR', i.meta)
                continue

    def clearQueue(self):
        self.ent_queue = None

    def executeFn(self, template):
        for i in template:
            if '.' in i:
                _ = i.split('.')
                fn = _[0]
                param = _[1]
            else:
                fn = i
                param = None

            res = self.template_fns[fn](param)

        if isinstance(res, dict):
            return res
        else:
            # return unique questions only
            return list({x['question']: x for x in res}.values())

    def thresholdSize(self, *args):
        def getSize(bbox):
            try:
                return (bbox['max'][0] - bbox['min'][0]) * (
                    bbox['max'][1] - bbox['min'][1]) * (
                        bbox['max'][2] - bbox['min'][2])
            except:
                return np.prod(bbox['radii']) * 8

        assert self.ent_queue != None
        assert self.ent_queue['type'] == 'objects'

        ent = self.ent_queue
        sizes = [getSize(x.bbox) for x in ent['elements']]
        idx = [i for i, v in enumerate(sizes) if v < 0.0005]

        for i in idx[::-1]:
            del ent['elements'][i]

        self.ent_queue = ent
        return self.ent_queue

    def blacklist(self, *args):
        assert self.ent_queue != None

        ent = self.ent_queue

        if ent['type'] == 'objects':
            template = args[0]

            names = [x.name for x in ent['elements']]
            idx = [
                i for i, v in enumerate(names)
                if v in self.blacklist_objects[template]
            ]
            for i in idx[::-1]:
                del ent['elements'][i]
        elif ent['type'] == 'rooms':
            names = [x.name for x in ent['elements']]
            idx = [
                i for i, v in enumerate([
                    any([k for k in x if k in self.blacklist_rooms])
                    for x in names
                ]) if v == True
            ]
            for i in idx[::-1]:
                del ent['elements'][i]

        self.ent_queue = ent
        return self.ent_queue

    def filter(self, *args):
        # if ent_queue is empty, execute on parent env entitites
        if self.ent_queue == None:
            self.ent_queue = {
                'type': args[0],
                'elements': self.entities[args[0]]
            }
        else:
            ent = self.ent_queue
            assert args[0] != ent['type']

            ent = {
                'type':
                args[0],
                'elements':
                [z for y in [x.entities for x in ent['elements']] for z in y]
            }
            self.ent_queue = ent

        # remove blacklisted rooms
        if self.ent_queue['type'] == 'rooms' and self.use_blacklist == True:
            self.ent_queue = self.blacklist()

        if self.ent_queue['type'] == 'objects' and self.use_threshold_size == True:
            self.ent_queue = self.thresholdSize()

        return self.ent_queue

    def unique(self, *args):
        assert self.ent_queue != None
        ent = self.ent_queue

        # unique based on room+object tuple
        if args[0] == 'combo':
            # self.ent_queue contains a list of objects
            names = [
                x.name + " IN " + "_".join(x.rooms[0].name)
                for x in ent['elements']
            ]

            idx = [
                i for i, v in enumerate([names.count(x) for x in names]) if v != 1
            ]
            for i in idx[::-1]:
                del ent['elements'][i]

            self.ent_queue = ent
            return self.ent_queue

        # unique based on either rooms or objects (only)
        names = [x.name for x in ent['elements']]
        idx = [
            i for i, v in enumerate([names.count(x) for x in names]) if v != 1
        ]

        for i in idx[::-1]:
            del ent['elements'][i]

        names = [x.name for x in ent['elements']]
        self.ent_queue = ent
        return self.ent_queue

    def query(self, *args):
        assert self.ent_queue != None
        ent = self.ent_queue

        return self.query_fns['query_' + args[0]](ent)

    def relate(self, *args):
        ent = self.ent_queue
        if len(ent['elements']) == 0:
            return ent

        if ent['type'] == 'objects':
            h_threshold, v_threshold = 0.05, 0.05
        elif ent['type'] == 'rooms':
            h_threshold, v_threshold = 5.0, 5.0
        nearby_object_pairs = self.house.getNearbyPairs(
            ent['elements'], hthreshold=h_threshold, vthreshold=v_threshold)

        self.ent_queue['elements'] = []
        for prep in ['on', 'next_to']:
            for el in nearby_object_pairs[prep]:
                if len([
                        x for x in nearby_object_pairs[prep]
                        if x[0].name == el[0].name
                ]) > 1:
                    continue

                if prep == 'on':
                    if el[2] > v_threshold / 1000.0:
                        preps = [('above', 1), ('under', 0)]
                    else:
                        preps = [('on', 1), ('below', 0)]
                elif prep == 'next_to':
                    preps = [('next_to', 0), ('next_to', 1)]

                self.ent_queue['elements'].append([el, preps])

        return self.ent_queue

    # only works with objectEntities for now
    def distance(self, *args):
        ent = self.ent_queue
        if ent['type'] == 'objects':
            h_low_threshold, h_high_threshold = 0.2, 2.0
        pairwise_distances = self.house.getAllPairwiseDistances(
            ent['elements'])

        # self.ent_queue['elements'] = []
        updated_ent_queue = {'type': ent['type'], 'elements': []}
        for i in ent['elements']:
            sub_list = [
                x for x in pairwise_distances
                if x[0].meta['id'] == i.meta['id']
                or x[1].meta['id'] == i.meta['id']
            ]
            sub_list = [
                x for x in sub_list if x[0].rooms[0].name == x[1].rooms[0].name
            ]
            far = [x for x in sub_list if x[2] >= h_high_threshold]
            close = [x for x in sub_list if x[2] <= h_low_threshold]
            if len(far) == 0 or len(close) == 0:
                continue
            for j in far:
                far_ent = 1 if j[0].name == i.name else 0
                for k in close:
                    close_ent = 1 if k[0].name == i.name else 0
                    updated_ent_queue['elements'].append(
                        [k[close_ent], i, j[far_ent], 'closer'])
                    updated_ent_queue['elements'].append(
                        [j[far_ent], i, k[close_ent], 'farther'])

        self.ent_queue = updated_ent_queue
        return self.ent_queue

    def queryRoom(self, ent):
        qns = []
        for i in ent['elements']:
            if len(i.rooms[0].name) > 1:
                if self.debug == True:
                    print('exception in queryRoom. room has multiple names.',
                          i.rooms[0].name)
                continue
            elif i.rooms[0].name == []:
                if self.debug == True:
                    print('exception in queryRoom. room has no name.', i.name,
                          i.rooms[0].name)
                continue
            if "_".join(i.rooms[0].name[0].split()) not in self.blacklist_rooms:
                qns.append(self.q_obj_builder('location', [i], i.rooms[0].name[0]))
        return qns

    def queryCount(self, ent):
        qns = []
        for i in ent['elements']:
            if len(i.rooms[0].name) > 1:
                if self.debug == True:
                    print('exception in queryCount. room has multiple names.',
                          i.rooms[0].name)
                continue
            elif i.rooms[0].name == []:
                if self.debug == True:
                    print('exception in queryCount. room has no name.', i.name,
                          i.rooms[0].name)
                continue

            count = len([x for x in i.rooms[0].objects if x.name == i.name])
            if count <= 5:
                qns.append(
                    self.q_obj_builder(
                        'count',
                        [x for x in i.rooms[0].objects
                         if x.name == i.name], count))
        return qns

    def queryRoomCounts(self, ent):
        qns = []
        rooms_done = set()

        # print [i.name for i in ent['elements']]
        exp_rooms = [
            name for room_ent in ent['elements'] for name in room_ent.name
        ]
        for i in ent['elements']:
            if i.name == []:
                if self.debug == True:
                    print('exception in queryRoomCount. room has no name.',
                          i.name, i.name)
                continue

            for room_name in i.name:
                if room_name in rooms_done: continue
                count = exp_rooms.count(room_name)
                # so that the correct room name is displayed in the question string
                i.name[0] = room_name
                if count < 5:
                    qns.append(
                        self.q_obj_builder('room_count', [
                            room_ent for room_ent in ent['elements']
                            if room_name in room_ent.name
                        ], count))
                rooms_done.add(room_name)
            # count = len([x for x in ent['elements'] if len(x.name) == 1 and x.name[0] == i.name[0]])

        return qns

    def queryRoomObjectCounts(self, ent):
        qns = []
        obj_to_room_names, obj_to_room_bbox = dict(), dict()
        for i in ent['elements']:
            # we should also include objects appearing in rooms
            # with multiple or no names (agent can walk through them)

            obj_name = i.name
            obj_room_bbox = i.rooms[0].meta['bbox']

            if len(i.rooms[0].name) == 0: room_name_for_obj = "none"
            elif len(i.rooms[0].name) > 1:
                room_name_for_obj = " ".join(i.rooms[0].name)
            else:
                room_name_for_obj = i.rooms[0].name[0]

            # update the room info for the obj. this update should be done only
            # if we have found an instance of the object in a new room (check using bbox dict)
            if obj_name not in obj_to_room_bbox:
                obj_to_room_bbox[obj_name] = []
            if obj_name not in obj_to_room_names:
                obj_to_room_names[obj_name] = []

            if obj_room_bbox not in obj_to_room_bbox[obj_name]:
                obj_to_room_bbox[obj_name].append(obj_room_bbox)
                obj_to_room_names[obj_name].append(room_name_for_obj)

        for obj_name in obj_to_room_names:
            ans = len(obj_to_room_names[obj_name])
            gt_bboxes = obj_to_room_bbox[obj_name]
            if ans <= 5:
                qns.append(
                    self.q_obj_builder(
                        # abusing notation here : the bbox entry for the "dummy"
                        # object entity is actually a list of bbox entries of the
                        # rooms where this object occurs in the house
                        'room_object_count',
                        [objectEntity(obj_name, gt_bboxes, {})],
                        ans))

        return qns

    def queryGlobalObjectCounts(self, ent):
        qns = []
        room_wise_dist = dict()
        rooms = []

        for i in ent['elements']:
            # Ignore objects which occur in rooms with no name or multiple names
            if len(i.rooms[0].name) > 1:
                if self.debug == True:
                    print('exception in queryCount. room has multiple names.',
                          i.rooms[0].name)
                continue
            elif i.rooms[0].name == []:
                if self.debug == True:
                    print('exception in queryCount. room has no name.', i.name,
                          i.rooms[0].name)
                continue

            room_name_for_obj = i.rooms[0].name[0]

            rooms.append(i.rooms[0])

            if room_name_for_obj not in room_wise_dist:
                room_wise_dist[room_name_for_obj] = []
            entities_in_room = room_wise_dist[room_name_for_obj]
            entities_in_room.append(i)
            room_wise_dist[room_name_for_obj] = entities_in_room

        for room_name in room_wise_dist:
            if room_name in self.blacklist_rooms: continue
            obj_entities = room_wise_dist[room_name]
            obj_names = [obj.name for obj in obj_entities]

            objs_done = set()
            for obj_entity in obj_entities:
                if obj_entity.name in objs_done: continue
                ans = obj_names.count(obj_entity.name)
                if ans <= 5:
                    qns.append(
                        self.q_obj_builder('global_object_count', [obj_entity],
                                           ans))
                objs_done.add(obj_entity.name)

        return qns

    def queryExist(self, ent):
        qns = []
        for i in ent['elements']:
            if len(i.rooms[0].name) > 1:
                if self.debug == True:
                    print('exception in queryExist. room has multiple names.',
                          i.rooms[0].name)
                continue
            elif i.rooms[0].name == []:
                if self.debug == True:
                    print('exception in queryExist. room has no name.', i.name,
                          i.rooms[0].name)
                continue

            qns.append(
                self.q_obj_builder(
                    'exist', [i], 'yes', q_type='exist_positive'))

            # generate list of object names in i.rooms[0].name in current env
            obj_present = [
                x.name for x in ent['elements']
                if len(x.rooms[0].name) != 0
                and x.rooms[0].name[0] == i.rooms[0].name[0]
            ]
            if i.rooms[0].name[0] not in self.negative_exists:
                self.negative_exists[i.rooms[0].name[0]] = []

            # generate list of object names for i.rooms[0].name not in i.rooms[0].name in current env
            obj_not_present = [
                x for x in self.global_obj_by_room[i.rooms[0].name[0]]
                if x[0] not in obj_present
                and x[0] not in self.negative_exists[i.rooms[0].name[0]]
            ]

            # create object entity and generate a no question
            if len(obj_not_present) == 0:
                continue

            self.negative_exists[i.rooms[0].name[0]].append(
                obj_not_present[0][0])
            sampled_obj = objectEntity(obj_not_present[0][0], {}, {})
            sampled_obj.addRoom(i.rooms[0])
            qns.append(
                self.q_obj_builder(
                    'exist', [sampled_obj], 'no', q_type='exist_negative'))

        return qns

    def queryLogical(self, ent):
        qns = []
        rooms_done = set()

        # the entities queue contains a list of object entities
        for i in ent['elements']:
            # ignore objects with (1) multiple and (2) no room names
            if len(i.rooms[0].name) > 1:
                if self.debug == True:
                    print(
                        'exception in queryLogical. room has multiple names.',
                        i.rooms[0].name)
                continue
            elif i.rooms[0].name == []:
                if self.debug == True:
                    print('exception in queryLogical. room has no name.',
                          i.name, i.rooms[0].name)
                continue

            if i.rooms[0].name[0] in rooms_done: continue
            # get list of all objects present in the same as room as the current object
            # note that as we iterate throgh the ent queue, all the objects in the same room
            # will generate identical list -- so we save the rooms processed in the room_done set

            # For example : if the first obj is a bed inside a bedroom, and this bedroom has
            # a total of 5 objects= : ['chair', 'bed', 'chair', 'dressing_table', 'curtains']
            # Then, whenever any of these objects is encountered in the loop (for i in ent['elements'])
            # we will end up generating the same list as shown
            local_list = [(x, x.name) for x in ent['elements']
                          if len(x.rooms[0].name) == 1
                          and x.rooms[0].name[0] == i.rooms[0].name[0]]
            local_objects_list_ = [obj for (obj, _) in local_list]
            local_object_names_list = [name for (_, name) in local_list]

            # get list of objects which are not present in the room where i resides.
            # this list is also pruned based on frequency
            # again, this list will be identical for all objects in the same room
            objs_not_present = [
                x[0] for x in self.global_obj_by_room[i.rooms[0].name[0]]
                if x[0] not in local_object_names_list
            ]

            both_present, both_absent, only_one_present = [], [], []
            # print ("Room : %s" % i.rooms[0].name)

            # populate objects for yes answer questions
            for i_idx in range(len(local_object_names_list)):
                for j_idx in range(i_idx + 1, len(local_object_names_list)):
                    if local_object_names_list[
                            i_idx] == local_object_names_list[j_idx]:
                        continue
                    both_present.append((local_object_names_list[i_idx],
                                         local_object_names_list[j_idx]))

            # populate objects for no answer questions -- part 1
            for i_idx in range(len(objs_not_present)):
                for j_idx in range(i_idx + 1, len(objs_not_present)):
                    if objs_not_present[i_idx] == objs_not_present[j_idx]:
                        continue
                    both_absent.append((objs_not_present[i_idx],
                                        objs_not_present[j_idx]))

            # populate objects for no answer questions -- part 2
            for obj1 in local_object_names_list:
                for obj2 in objs_not_present:
                    only_one_present.append((obj1, obj2))

            # generate a question for each object pairs in the 3 lists
            shuffle(both_present)
            shuffle(both_absent)
            shuffle(only_one_present)
            num_yes = num_no = len(both_present)
            only_one_present, both_absent = only_one_present[:int(
                num_no - num_no / 2)], both_absent[:int(num_no / 2)]

            for (obj1, obj2) in both_present:
                obj1_entity, obj2_entity = objectEntity(obj1, {},
                                                        {}), objectEntity(
                                                            obj2, {}, {})
                obj1_entity.rooms.append(i.rooms[0])
                qns.append(
                    self.q_obj_builder('exist_logic',
                                       [(obj1_entity, obj2_entity)], 'yes',
                                       'exist_logical_positive'))
                qns.append(
                    self.q_obj_builder('exist_logic',
                                       [(obj1_entity, obj2_entity)], 'yes',
                                       'exist_logical_or_positive_1'))
            for (obj1, obj2) in both_absent:
                obj1_entity, obj2_entity = objectEntity(obj1, {},
                                                        {}), objectEntity(
                                                            obj2, {}, {})
                obj1_entity.rooms.append(
                    i.rooms[0]
                )  # this is not technically correct, just so that q_string_builder works
                qns.append(
                    self.q_obj_builder('exist_logic',
                                       [(obj1_entity, obj2_entity)], 'no',
                                       'exist_logical_negative_1'))
                qns.append(
                    self.q_obj_builder('exist_logic',
                                       [(obj1_entity, obj2_entity)], 'no',
                                       'exist_logical_or_negative'))

            for (obj1, obj2) in only_one_present:
                obj1_entity, obj2_entity = objectEntity(obj1, {},
                                                        {}), objectEntity(
                                                            obj2, {}, {})
                obj1_entity.rooms.append(i.rooms[0])
                qns.append(
                    self.q_obj_builder('exist_logic',
                                       [(obj1_entity, obj2_entity)], 'no',
                                       'exist_logical_negative_2'))
                qns.append(
                    self.q_obj_builder('exist_logic',
                                       [(obj1_entity, obj2_entity)], 'yes',
                                       'exist_logical_or_positive_2'))

            # mark room as done
            rooms_done.add(i.rooms[0].name[0])

        return qns

    def queryColor(self, ent):
        qns = []
        for i in ent['elements']:
            if self.house.id + '.' + i.id in  self.env_obj_color_map:
                color =  self.env_obj_color_map[self.house.id + '.' + i.id]
                qns.append(
                    self.q_obj_builder('color', [i], color))
            else:
                # no color
                continue
        return qns

    def queryColorRoom(self, ent):
        qns = []
        for i in ent['elements']:
            if len(i.rooms[0].name) > 1:
                if self.debug == True:
                    print('exception in queryExist. room has multiple names.',
                          i.rooms[0].name)
                continue
            elif i.rooms[0].name == []:
                if self.debug == True:
                    print('exception in queryExist. room has no name.', i.name,
                          i.rooms[0].name)
                continue

            if self.house.id + '.' + i.id in  self.env_obj_color_map:
                color =  self.env_obj_color_map[self.house.id + '.' + i.id]
                qns.append(
                    self.q_obj_builder('color_room', [i], color))
            else:
                # no color
                continue
        return qns

    def queryObject(self, ent):
        qns = []
        for i in ent['elements']:
            el = i[0]
            preps = i[1]
            for prep_mod in preps:
                if el[prep_mod[1] ^ 1].name not in self.blacklist_objects['relate']:
                    qns.append(
                        self.q_obj_builder(prep_mod[0], [el[prep_mod[1]]],
                                           el[prep_mod[1] ^ 1].name))
        return qns

    def queryObjectRoom(self, ent):
        qns = []
        for i in ent['elements']:
            el = i[0]
            preps = i[1]
            if len(el[0].rooms[0].name) > 1:
                if self.debug == True:
                    print('exception in queryExist. room has multiple names.',
                          el[0].rooms[0].name)
                continue
            elif el[0].rooms[0].name == []:
                if self.debug == True:
                    print('exception in queryExist. room has no name.',
                          el[0].name, el[0].rooms[0].name)
                continue
            for prep_mod in preps:
                if el[prep_mod[1] ^ 1].name not in self.blacklist_objects['relate']:
                    qns.append(
                        self.q_obj_builder(prep_mod[0] + '_room',
                                           [el[prep_mod[1]]],
                                           el[prep_mod[1] ^ 1].name))
        return qns

    def queryCompare(self, ent):
        qns = []
        for i in ent['elements']:
            if len(i[0].rooms[0].name) > 1:
                if self.debug == True:
                    print('exception in queryExist. room has multiple names.',
                          i[0].rooms[0].name)
                continue
            elif i[0].rooms[0].name == []:
                if self.debug == True:
                    print('exception in queryExist. room has no name.',
                          i[0].name, i[0].rooms[0].name)
                continue
            qns.append(
                self.q_obj_builder(i[3] + '_room', i[:3], 'yes',
                                   'dist_compare_positive'))
            qns.append(
                self.q_obj_builder(i[3] + '_room', i[:3][::-1], 'no',
                                   'dist_compare_negative'))
        return qns

    def questionObjectBuilder(self, template, q_ent, a_str, q_type=None):
        if q_type == None:
            q_type = template

        q_str = self.templates[template]
        bbox = []

        if template == 'room_count':
            # if this condition holds, the question type is 'room_count' and the q_ent[0] is a room entity
            q_str = self.q_str_builder.prepareString(q_str, '',
                                                     q_ent[0].name[0])
            return {
                'question':
                q_str,
                'answer':
                a_str,
                'type':
                q_type,
                'meta': {},
                'bbox': [{
                    'type': x.type,
                    'box': x.bbox,
                    'name': x.name,
                    'target': True
                } for x in q_ent]
            }

        if template == 'room_object_count':
            q_str = self.q_str_builder.prepareString(q_str, q_ent[0].name, '')
            return {
                'question':
                q_str,
                'answer':
                a_str,
                'type':
                q_type,
                'meta': {},
                'bbox': [{
                    'type': x.type,
                    'box': x.bbox,
                    'name': x.name,
                    'target': True
                } for x in q_ent]
            }

        if template == 'global_object_count':
            # if (len(q_ent) == 1) and (not isinstance(q_ent[0], tuple)) and (q_ent[0].type == 'object'):
            # if this condition holds, the question type is 'global_object_count' and the q_ent[0] is an obj entity
            q_str = self.q_str_builder.prepareString(q_str, q_ent[0].name,
                                                     q_ent[0].rooms[0].name[0])
            return {
                'question': q_str,
                'answer': a_str,
                'type': q_type,
                'meta': {},
                'bbox': [{}]
            }

        for ent in q_ent:
            # if ent is a tuple, it means exist_logic questions
            if isinstance(ent, tuple):
                if 'or' in q_type:
                    q_str = self.q_str_builder.prepareStringForLogic(
                        q_str, ent[0].name, ent[1].name,
                        ent[0].rooms[0].name[0], "or")
                else:
                    q_str = self.q_str_builder.prepareStringForLogic(
                        q_str, ent[0].name, ent[1].name,
                        ent[0].rooms[0].name[0], "and")
                return {
                    'question':
                    q_str,
                    'answer':
                    a_str,
                    'type':
                    q_type,
                    'meta': {},
                    'bbox': [{
                        'type': ent[0].rooms[0].type,
                        'box': ent[0].rooms[0].bbox,
                        'name': ent[0].rooms[0].name,
                        'target': True
                    }]
                }

            bbox.append({
                'type': ent.type,
                'box': ent.bbox,
                'name': ent.name,
                'target': True
            })
            if not isinstance(ent, tuple) and len(ent.rooms[0].name) != 0:
                q_str = self.q_str_builder.prepareString(
                    q_str, ent.name, ent.rooms[0].name[0])
            else:
                q_str = self.q_str_builder.prepareString(q_str, ent.name, '')

            if not isinstance(ent, tuple):
                if len(ent.rooms[0].name) == 0:
                    name = []
                else:
                    name = ent.rooms[0].name
                bbox.append({
                    'type': ent.rooms[0].type,
                    'box': ent.rooms[0].bbox,
                    'name': name,
                    'target': False
                })

        if 'mat' in q_ent[0].meta:
            mat = q_ent[0].meta['mat']
        else:
            mat = {}

        return {
            'question': q_str,
            'answer': a_str,
            'type': q_type,
            'meta': mat,
            'bbox': bbox
        }



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-inputJson",
        help="Environments list",
        default=
        "data/envs.json"
    )
    parser.add_argument(
        "-outputJson",
        help="Json file to save questions list",
        default="data/working/questions.json")
    parser.add_argument(
        "-dataDir",
        help="Directory that has the SUNCG dataset",
        default="/path/to/suncg/")
    parser.add_argument(
        "-numEnvs", help="No. of environments to read", default=False)
    args = parser.parse_args()

    data = json.load(open(args.inputJson, 'r'))
    house_ids = list(data.keys())
    num_envs = len(house_ids)
    if args.numEnvs != False:
        num_envs = int(args.numEnvs)

    Hp = HouseParse(dataDir=args.dataDir)
    E = Engine()

    # SAVE QUESTIONS TO A JSON FILE
    #
    # -- STARTS HERE

    idx, all_qns = 1, []
    empty_envs = []
    for i in tqdm(range(num_envs)):
        Hp.parse(house_ids[i])
        num_qns_for_house = 0
        for j in E.template_defs:
            if j == 'global_object_count': continue
            E.cacheHouse(Hp)
            qns = E.executeFn(E.template_defs[j])
            num_qns_for_house += len(qns)
            E.clearQueue()

            for k in qns:
                k['id'] = idx
                k['house'] = house_ids[i]
                idx += 1

                all_qns.append(k)

        if num_qns_for_house == 0: empty_envs.append(house_ids[i])

    print("Houses with no questions generated (if any) : %d" % len(empty_envs))
    print(empty_envs)

    print('Writing to %s...' % args.outputJson)
    json.dump(all_qns, open(args.outputJson, 'w'))

    # -- ENDS HERE

    # TO GET ENVS WHICH CONTAIN INSTANCES OF A SPECIFIC OBJ TYPE
    #
    # -- STARTS HERE

    # envs_with_obj = dict()
    # for i in tqdm(range(num_envs)):
    #     Hp.parse(house_ids[i])
    #     E.cacheHouse(Hp)
    #     if house_ids[i] not in envs_with_obj: envs_with_obj[house_ids[i]] = 0

    #     res = E.executeFn(['filter.objects'])
    #     for j in res['elements']:
    #         if j.name == 'book':
    #             envs_with_obj[house_ids[i]] += 1

    #     E.clearQueue()
    # env_list = [env for env in envs_with_obj if envs_with_obj[env] != 0]
    # ans = len(env_list)
    # for env in env_list: print env
    # print ("There are %d envs with 'tabel_and_chair' instances..." % ans)

    # -- ENDS HERE

    # PRINT QUESTION COUNTS FOR ALL ENVS
    #
    # -- STARTS HERE

    # for i in range(num_envs):
    #     Hp.parse(house_ids[i])
    #     E.cacheHouse(Hp)
    #     count = {}
    #     for j in E.template_defs:
    #         qns = E.executeFn(E.template_defs[j])
    #         E.clearQueue()
    #         for k in qns:
    #             if k['type'] not in count:
    #                 count[k['type']] = []
    #             count[k['type']].append(k)
    #     total = 0
    #     print('---| %s' % house_ids[i])
    #     for j in count:
    #         print('%30s: %5d' % (j, len(count[j])))
    #         total += len(count[j])
    #     print('%30s: %5d' % ('total', total))

    # -- ENDS HERE

    # TO GENERATE OBJECT COUNTS BY ROOM FOR ALL ENVS
    #
    # -- STARTS HERE

    # object_counts_by_room = {}
    # for i in tqdm(range(num_envs)):
    #     Hp.parse(house_ids[i])
    #     E.cacheHouse(Hp)

    #     res = E.executeFn(['filter.rooms', 'filter.objects', 'blacklist.exist'])

    #     for j in res['elements']:
    #         for k in j.rooms[0].name:
    #             if k not in object_counts_by_room:
    #                 object_counts_by_room[k] = {}
    #             if j.name not in object_counts_by_room[k]:
    #                 object_counts_by_room[k][j.name] = 0
    #             object_counts_by_room[k][j.name] += 1

    #     E.clearQueue()
    # for i in object_counts_by_room:
    #     object_counts_by_room[i] = sorted(
    #         object_counts_by_room[i].items(), key=lambda x: x[1], reverse=True)
    # json.dump(object_counts_by_room, open('data/obj_counts_by_room.json', 'w'))

    # -- ENDS HERE

    # GATHERING GLOBAL STATISTICS BY QUESTION TYPE
    #
    # -- STARTS HERE

    # global_counts_by_q_type = {'total': []}
    # for i in range(num_envs):
    #     Hp.parse(house_ids[i])
    #     E.cacheHouse(Hp)
    #     count = {}
    #     for j in E.template_defs:
    #         qns = E.executeFn(E.template_defs[j])
    #         E.clearQueue()
    #         for k in qns:
    #             if k['type'] not in count:
    #                 count[k['type']] = []
    #             count[k['type']].append(k)
    #     total = 0
    #     print('---| %s' % house_ids[i])
    #     for j in count:
    #         print('%30s: %5d' % (j, len(count[j])))
    #         total += len(count[j])
    #         if j not in global_counts_by_q_type:
    #             global_counts_by_q_type[j] = []
    #         global_counts_by_q_type[j].append(len(count[j]))
    #     global_counts_by_q_type['total'].append(total)
    #     print('%30s: %5d' % ('total', total))

    # print('---| GLOBAL STATS ACROSS %d ENVS' % len(global_counts_by_q_type['total']))
    # for i in global_counts_by_q_type:
    #     print('%30s, mean: %.03f var: %.03f' % (i, np.array(global_counts_by_q_type[i]).mean(), np.array(global_counts_by_q_type[i]).std()))

    # -- ENDS HERE
