# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Parses SunCG json files to generate object+attribute lists.
"""

import csv
import argparse
import os, sys, json

import numpy as np


class HouseParse():
    def __init__(
            self,
            dataDir):
        self.dataDir = dataDir

        csvFile = csv.reader(
            open('../../House3D/House3D/metadata/ModelCategoryMapping.csv', 'r'))
        headers = next(csvFile)

        self.modelCategoryMapping = {}

        for row in csvFile:
            self.modelCategoryMapping[row[headers.index('model_id')]] = {
                headers[x]: row[x]
                for x in range(2, len(headers))  # 0 is index, 1 is model_id
            }

    """
    Go over all nodes of house environment and accumulate objects room-wise.
    Optionally, filter out objects close to the edge of the house (might be
    inaccessible) or some boring "objects" like "plants".
    """

    def parse(self, houseId, levelsToExplore=[0]):
        self.id = houseId
        jsonPath = os.path.join(self.dataDir, 'house', houseId, 'house.json')
        data = json.load(open(jsonPath, 'r'))

        self.houseBBox = data['bbox']

        self.rooms, self.objects = [], {}
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

                    roomType = [' '.join(x.lower().split('_')) for x in data['levels'][i]['nodes'][j]['roomTypes']]

                    nodes = data['levels'][i]['nodes'][j][
                        'nodeIndices'] if 'nodeIndices' in data['levels'][i][
                            'nodes'][j] else []
                    self.rooms.append({
                        'type':
                        roomType,
                        'bbox':
                        data['levels'][i]['nodes'][j]['bbox'],
                        'nodes':
                        nodes
                    })

                # Objects
                elif data['levels'][i]['nodes'][j]['type'] == 'Object':
                    if 'materials' not in data['levels'][i]['nodes'][j]:
                        material = []
                    else:
                        material = data['levels'][i]['nodes'][j]['materials']
                    self.objects[data['levels'][i]['nodes'][j]['id']] = {
                        'id':
                        data['levels'][i]['nodes'][j]['id'],
                        'fine_class':
                        self.modelCategoryMapping[data['levels'][i]['nodes'][
                            j]['modelId']]['fine_grained_class'],
                        'coarse_class':
                        self.modelCategoryMapping[data['levels'][i]['nodes'][
                            j]['modelId']]['coarse_grained_class'],
                        'bbox':
                        data['levels'][i]['nodes'][j]['bbox'],
                        'mat':
                        material
                    }

    """
    Returns a list of object id pairs that are close together
    for use with next to / on / below question template

    Takes object keys as parameter; keys to index into the self.objects dict

    next to
    -------
    - horizontal distance should be below a threshold
    - horizontal extent of one should not be contained within other
    - vertical extent of one should be contained within other

    on / below
    ----------
    - vertical distance should be below a threshold
    - vertical extent of one should not be contained within other
    - horizontal extent of one should be contained within other

    """
    def getNearbyPairs(self,
                             availableEnts,
                             hthreshold=0.005,
                             vthreshold=0.005):
        assert len(availableEnts) != 0

        nearbyPairs = {'on': [], 'next_to': []}

        for i in range(len(availableEnts) - 1):
            for j in range(i + 1, len(availableEnts)):
                if availableEnts[i].name != availableEnts[j].name:
                    # next to
                    dist = self.getClosestDistance(
                        availableEnts[i].meta,
                        availableEnts[j].meta,
                        axes=[0, 2])
                    if dist < hthreshold and (self.isContained(
                            availableEnts[i].meta,
                            availableEnts[j].meta,
                            axis=0) & self.isContained(
                                availableEnts[i].meta,
                                availableEnts[j].meta,
                                axis=2)) == False:
                        if availableEnts[i].type == 'room' and availableEnts[j].type == 'room':
                            nearbyPairs['next_to'].append(
                                (availableEnts[i], availableEnts[j], dist))
                        elif self.isContained(
                            availableEnts[i].meta,
                            availableEnts[j].meta,
                            axis=1) == True:
                            nearbyPairs['next_to'].append(
                                (availableEnts[i], availableEnts[j], dist))

                    # on / below
                    if availableEnts[i].type == 'object' and availableEnts[j].type == 'object':
                        dist = self.getClosestDistance(
                            availableEnts[i].meta,
                            availableEnts[j].meta,
                            axes=[1])
                        if dist < vthreshold and (self.isContained(
                                availableEnts[i].meta,
                                availableEnts[j].meta,
                                axis=0) & self.isContained(
                                    availableEnts[i].meta,
                                    availableEnts[j].meta,
                                    axis=2)) == True and self.isContained(
                                        availableEnts[i].meta,
                                        availableEnts[j].meta,
                                        axis=1) == False:

                            # higher first
                            if self.isHigher(
                                    availableEnts[i].meta,
                                    availableEnts[j].meta,
                                    axis=1):
                                nearbyPairs['on'].append(
                                    (availableEnts[i], availableEnts[j], dist))
                            elif self.isHigher(
                                    availableEnts[i].meta,
                                    availableEnts[j].meta,
                                    axis=1):
                                nearbyObjectPairs['on'].append(
                                    (availableEnts[j], availableEnts[i], dist))

        return nearbyPairs

    """
    Returns distance between closest corners of two objects with 'bbox' attribute

    axes. [0,2] for horizontal distance, [1] for vertical distance
    order. 2 for Euclidean, 1 for Manhattan, etc
    """

    def getClosestDistance(self, obj1, obj2, axes=[0, 2], order=2):
        assert 'bbox' in obj1 and 'bbox' in obj2

        bbox = [
            {
                'min': np.array(obj1['bbox']['min'])[axes],
                'max': np.array(obj1['bbox']['max'])[axes]
            },
            {
                'min': np.array(obj2['bbox']['min'])[axes],
                'max': np.array(obj2['bbox']['max'])[axes]
            },
        ]

        cornerInds = [[
            int(i) for i in list('{0:0{width}b}'.format(j, width=len(axes)))
        ] for j in range(2**len(axes))]
        corners = [[
            np.array(
                [bbox[obj][['min', 'max'][i[j]]][j] for j in range(len(axes))])
            for i in cornerInds
        ] for obj in range(2)]

        dist = 1e5
        for i in range(len(corners[0])):
            for j in range(len(corners[1])):
                d = np.linalg.norm(corners[0][i] - corners[1][j], ord=order)
                if d < dist:
                    dist = d

        return dist

    def getAllPairwiseDistances(self, objList, axes=[0, 2], order=2):
        objDistances, objDistancesHash = [], {}
        for i in objList:
            for j in objList:
                if i.name == j.name:
                    continue
                if i.name + 'x' + j.name in objDistances or j.name + 'x' + i.name in objDistancesHash:
                    continue
                else:
                    dist = self.getClosestDistance(
                        i.meta, j.meta, axes, order)
                    objDistances.append((i, j, dist))
                    objDistancesHash[i.name + 'x' + j.name] = True

        return sorted(objDistances, key=lambda x: -x[2])

    """
    Checks if coordinates along given axis of one object is contained in another
    obj1, obj2 should have 'bbox' attributes
    """

    def isContained(self, obj1, obj2, axis=0):
        if obj1['bbox']['min'][axis] < obj2['bbox']['min'][axis] and obj1['bbox']['max'][axis] > obj2['bbox']['max'][axis]:
            return True
        elif obj2['bbox']['min'][axis] < obj1['bbox']['min'][axis] and obj2['bbox']['max'][axis] > obj1['bbox']['max'][axis]:
            return True
        else:
            return False

    """
    Checks if obj1 is higher than obj2 along given axis
    """

    def isHigher(self, obj1, obj2, axis=0):
        if obj1['bbox']['max'][axis] > obj2['bbox']['max'][axis]:
            return True
        else:
            return False

    """
    Finds closest accessible point to given coordinates

    Make sure 'points' are rescaled (world.rescale())
    """

    def findClosestPoints(self, points, move_map, move_tree):
        res = []

        for el in points:
            dist, idx = move_tree.query([el[0], el[1]])
            res.append([dist, idx, move_map[idx]])

        return res

    def getVertices(self, bbox):
        x1, _, y1 = bbox['min']
        x2, _, y2 = bbox['max']

        return [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-houseId",
        help="SunCG House ID",
        default="00065ecbdd7300d35ef4328ffe871505")
    parser.add_argument(
        "-dataDir",
        help="Directory that has the SUNCG dataset",
        default="/Users/abhshkdz/data/suncg/")
    args = parser.parse_args()

    Hp = HouseParse(dataDir=args.dataDir)
    Hp.parse(args.houseId)

    rooms, objects = Hp.rooms, Hp.objects
    print ("There are %d rooms in the house" % len(rooms))

