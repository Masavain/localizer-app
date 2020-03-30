import struct, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from math import ceil
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
matplotlib.use('agg')
from itertools import cycle, islice
from os.path import join, dirname, realpath
import os

def get_image_features():
    with open('out.txt') as file:
        f = [[int(j.strip()) for j in line.strip('[;]\n\t ').split(',')] for line in file]
        image_features = [bytes(bytearray(i)) for i in f]
    return image_features
        
def read_cstring(file):
    c = file.read(1)
    while c[-1] != 0:
        c += file.read(1)
    return c[:-1]

def get_map_landmarks_keyframes():
    with open('map.db', 'rb') as file:
        read32i = lambda: int.from_bytes(file.read(4), 'little')
        read32f = lambda: struct.unpack('f', file.read(4))[0]
        read64f = lambda: struct.unpack('d', file.read(8))[0]
        
        frame_next_id = read32i()
        keyframe_next_id = read32i()
        landmark_next_id = read32i()
        camera_count = read32i()
        
        cameras = []
        for i in range(camera_count):
            camera = {}
            camera['camera_name'] = read_cstring(file)
            camera['model_type'] = read_cstring(file)
            camera['setup_type'] = read_cstring(file)
            camera['color_order'] = read_cstring(file)
            camera['rows'] = read32i()
            camera['cols'] = read32i()
            camera['fps'] = read64f()
            if camera['model_type'] == b"Fisheye":
                fx, fy, cx, cy, k1, k2, k3, k4, focal_x_baseline = struct.unpack('9d', file.read(9*8))
                camera['fx'] = fx
                camera['fy'] = fy
                camera['cx'] = cx
                camera['cy'] = cy
                camera['k1'] = k1
                camera['k2'] = k2
                camera['k3'] = k3
                camera['k4'] = k4
                camera['focal_x_baseline'] = focal_x_baseline
            elif camera['model_type'] == b"Perspective":
                fx, fy, cx, cy, k1, k2, p1, p2, k3, focal_x_baseline = struct.unpack('10d', file.read(10*9))
                camera['fx'] = fx
                camera['fy'] = fy
                camera['cx'] = cx
                camera['cy'] = cy
                camera['k1'] = k1
                camera['k2'] = k2
                camera['k3'] = k3
                camera['p1'] = k4
                camera['p2'] = k4
                camera['focal_x_baseline'] = focal_x_baseline
            elif camera['model_type'] == b"Equirectangular":
                # This might be a padding byte, but we're unlikely to ever run into it
                pass
            cameras.append(camera)
        keyframe_count = read32i()
        keyframes = []
        for i in range(keyframe_count):
            keyframe = {}
            keyframe['id'] = read32i()
            keyframe['depth_thr'] = read32f()
            keyframe['n_keypoints'] = read32i()
            keyframe['num_scale_levels'] = read32i()
            keyframe['scale_factor'] = read32f()
            keyframe['source_frame_id'] = read32i()
            keyframe['span_parent'] = read32i()
            keyframe['timestamp'] = read64f()
            
            keyframe['rot_x'] = read64f()
            keyframe['rot_y'] = read64f()
            keyframe['rot_z'] = read64f()
            keyframe['rot_w'] = read64f()
            keyframe['trans_x'] = read64f()
            keyframe['trans_y'] = read64f()
            keyframe['trans_z'] = read64f()
            
            keyframe['camera_name'] = cameras[read32i()]['camera_name']
            keyframe['depths'] = [read32f() for _ in range(read32i())]
            keyframe['descriptors'] = [file.read(32) for _ in range(read32i())]
            keyframe['landmark_ids'] = [read32i() for _ in range(read32i())]
            keyframe['loop_edge_ids'] = [read32i() for _ in range(read32i())]
            keyframe['spanning_child_ids'] = [read32i() for _ in range(read32i())]
            keyframe['x_rights'] = [read32i() for _ in range(read32i())]
            keyframe['keypoints'] = [{"x": read32f(), "y": read32f(), "angle": read32f(), "octave": read32i()} for _ in range(read32i())]
            keyframe['undistorted_keypoints'] = [{"x": read32f(), "y": read32f()} for _ in range(read32i())]
            
            keyframes.append(keyframe)
        landmarks = [{"id": read32i(), "first_keyframe_id": read32i(), "ref_keyframe_id": read32i(),
                    "num_observable": read32i(), "num_observed": read32i(), "padding": read32i(),
                    "pos_x": read64f(), "pos_y": read64f(), "pos_z": read64f()} for _ in range(read32i())]
        
        assert(file.read() == b'')
    return landmarks, keyframes



def localize_image():
    image_features = get_image_features()
    landmarks, keyframes = get_map_landmarks_keyframes()

    landmarks_by_id = {i["id"]: i for i in landmarks}


    # descriptors of map keyframes
    descriptors = {}

    # positions of keyframes
    keyframe_positions = []

    for keyframe in keyframes:
        keyframe_positions.append((keyframe['trans_x'], keyframe['trans_y'], keyframe['trans_z']))
        for descriptor, landmark_id in zip(keyframe['descriptors'], keyframe['landmark_ids']):
            if (landmark_id == 4294967295):
                continue
            if descriptor not in descriptors:
                descriptors[descriptor] = []
            landmark = landmarks_by_id[landmark_id]
            # add landmark positions that correspond to keyframes
            descriptors[descriptor].append((landmark['pos_x'], landmark['pos_y'], landmark['pos_z']))
            #descriptors[descriptor].append((keyframe['trans_x'], keyframe['trans_y'], keyframe['trans_z']))

    dsc_keys = list(descriptors.keys())
    im_array = np.array([[j for j in i] for i in image_features], dtype='uint8')
    map_array = np.array([[j for j in i] for i in dsc_keys], dtype='uint8')
    print('Image features:', im_array.shape)
    print('Map features:', map_array.shape)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    k = 200000
    c = ceil(map_array.shape[0] / k)
    matches = []
    for i in range(c):
        print("Matching", i+1, "/", c)
        matches.extend(matcher.match(im_array, map_array[i*k:(i+1)*k]))

    matches_orig = matches
    trainids_orig = [x.trainIdx for x in matches_orig]
    queryids_orig = [x.queryIdx for x in matches_orig]
    #print("train id:s of all matches")
    #print(len(trainids_orig))
    #print(len(set(trainids_orig)))

    #print("query id:s of all matches ")
    #print(len(queryids_orig))
    #print(len(set(queryids_orig)))

    #matches = filter(lambda x: x.distance < 50, matches)
    matches = sorted(matches, key=lambda x: x.distance)
    # print n. of Matches within distance of 50
    print('Matches:', len(matches))


    counts_trainids = dict((x.trainIdx, trainids_orig.count(x.trainIdx)) for x in matches_orig)
    counts_trainids = {k: v for k, v in sorted(counts_trainids.items(), key=lambda item: item[1], reverse=True)}
    #print(counts_trainids)

    counts_queryids = dict((x.queryIdx, queryids_orig.count(x.queryIdx)) for x in matches_orig)
    counts_queryids = {k: v for k, v in sorted(counts_queryids.items(), key=lambda item: item[1], reverse=True)}
    #print(counts_queryids)

    #matches = [x for x in matches_orig if counts_queryids[x.queryIdx]==1 and counts_trainids[x.trainIdx]==1]
    matches = matches_orig

    #matches = filter(lambda x: x.distance < 50, matches)
    #matches = sorted(matches, key=lambda x: x.distance)
    #matches = matches[:10]


    points = set()
    for i in descriptors.values():
        for j in i:
            points.add(j)
            
    xs, ys, zs = zip(*points)

    found_points = {}

    max_distance = max(x.distance for x in matches)
    min_distance = min(x.distance for x in matches)
    score_sum = 0



    matches = filter(lambda x: x.distance < 20, matches)
    matches = filter(lambda x: counts_queryids[x.queryIdx] == 5, matches)

    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:20]

    coordinates = [descriptors[dsc_keys[match.trainIdx]][0] for match in matches]
    coordinates = [[i[0],i[2]] for i in coordinates]



    clustering = KMeans(n_clusters=3,init='k-means++').fit(coordinates)

    y_pred = clustering.labels_.astype(np.int)

    colors = np.array(list(islice(cycle(['#dede00', '#a65628', '#4daf4a']), int(max(y_pred) + 1))))
        
    cluster_centers = clustering.cluster_centers_

    for match in matches:
        for i in descriptors[dsc_keys[match.trainIdx]]:
            #print(i)
            #print(counts_trainids[match.trainIdx])
            #print(counts_queryids[match.queryIdx])
            #print(match.distance)

            if i not in found_points:
                found_points[i] = 0
            found_points[i] += 1-(match.distance-min_distance)/(max_distance-min_distance)
            score_sum += 1-(match.distance-min_distance)/(max_distance-min_distance)
        
    sum_x, sum_y, sum_z = 0, 0, 0

    for i in found_points:
        sum_x += i[0] * found_points[i]
        sum_y += i[1] * found_points[i]
        sum_z += i[2] * found_points[i]
        
    avg_x = sum_x / score_sum
    avg_y = sum_y / score_sum
    avg_z = sum_z / score_sum

    xs_f, ys_f, zs_f = zip(*found_points.keys())
    s = found_points.values()

    xk, yk, zk = zip(*keyframe_positions)
    
    plt.xlim(-125, 125)
    plt.ylim(-150, 150)
    plt.scatter(xs, zs, 0.1)
    plt.scatter(xs_f, zs_f, color=colors[y_pred])
    #plt.scatter(xs_f, zs_f, list([(i+0.1)*25 for i in s]))
    #plt.scatter(avg_x, avg_z, color='m')
    plt.scatter(cluster_centers[:,0], cluster_centers[:,1])
    #plt.scatter(xk, zk)
    print(dirname(realpath(__file__)))
    path = os.path.join(dirname(realpath(__file__)),'plots/plot.png')
    print(path)
    plt.savefig(path, format='png')
    
    #plt.show()
