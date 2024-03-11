# 应用于双臂，
# 抓取点1：鲜红色（1,0,0）
# 抓取点2：橙红色（1,80/255,5/255）/粉色(245,5,140)/黄色（232,249,1）
# 放置点1：纯蓝色（0,0,1）
# 放置点2：紫色（187/255,0,255/255）
# 其余位置颜色：浅天蓝色（0.58,0.8，1）/
# 备选（212/255,215/255,1）
# 设置半径超参为0.15

import numpy as np
import mitsuba as mi
import os
import matplotlib.pyplot as plt

ra1=0.2
rb2=0.3
rc3=0.1

def generate_random_number():
    while True:
        random_number = np.random.normal()
        if -20 <= random_number <= 20:
            return random_number

def generate_four_points_heatmap_xml_for_mitsuba(
        pts1,  # [num_points,3]
        feature1,  # [num_points,512]
        select_point1, select_point_feature1,
        select_point2, select_point_feature2,
        select_point3, select_point_feature3,
        select_point4, select_point_feature4,
        image_size={'xres': 640, 'yres': 640},
        camera_intrinsic={'fov': 20},
        camera_extrinsic={'location': "3,3,3",
                          'look_at': "0,0,0", 'up': "0,0,1"},
        light_extrinsic={'location': "-4,4,20",
                         'look_at': "0,0,0", 'up': "0,0,1"},
        path='/your/xml/save/path',
        xml_name1='one'): 
    def standardize_bbox(pcl, feature, points_per_object,
                        select_point1, select_point_feature1, 
                        select_point2, select_point_feature2,
                        select_point3, select_point_feature3,
                        select_point4, select_point_feature4):
        if pcl.shape[0] > points_per_object:
            pt_indices = np.random.choice(
                pcl.shape[0], points_per_object, replace=False)
            np.random.shuffle(pt_indices)
            pcl = pcl[pt_indices]  # n by 3

        mins = np.amin(pcl, axis=0)
        maxs = np.amax(pcl, axis=0)
        center = (mins + maxs) / 2.
        scale = np.amax(maxs-mins)
        print("Center: {}, Scale: {}".format(center, scale))
        result = ((pcl - center)/scale).astype(np.float32)  # [-0.5, 0.5]
        result[0]=((select_point1-center)/scale).astype(np.float32)
        result[1]=((select_point2-center)/scale).astype(np.float32)
        result[2]=((select_point3-center)/scale).astype(np.float32)
        result[3]=((select_point4-center)/scale).astype(np.float32)
        return result, feature, center, scale

    # feature[num_point,512]  point_feature[512]
    def colormap(feature, pts,
                select_point1, select_point_feature1, 
                select_point2, select_point_feature2,
                select_point3, select_point_feature3,
                select_point4, select_point_feature4):
        color_map=np.zeros((pts.shape[0],3),dtype=np.float32)
        limit1 = np.random.uniform(ra1,rb2)
        limit2 = np.random.uniform(ra1,rb2)
        limit3 = np.random.uniform(ra1,rb2)
        limit4 = np.random.uniform(ra1,rb2)
        for i in range(pts.shape[0]):
            distance1 = np.clip(np.linalg.norm(pts[i] - select_point1)+generate_random_number()/180,0,1)
            distance2 = np.clip(np.linalg.norm(pts[i] - select_point2)+generate_random_number()/180,0,1)
            distance3 = np.clip(np.linalg.norm(pts[i] - select_point3)+generate_random_number()/130,0,1)
            distance4 = np.clip(np.linalg.norm(pts[i] - select_point4)+generate_random_number()/130,0,1)
            a=np.random.uniform(rc3, limit1)
            b=np.random.uniform(rc3, limit2)
            c=np.random.uniform(rc3, limit3)
            d=np.random.uniform(rc3, limit4)
            if distance1< a:
                color_map[i]=np.array([1.0,0.0,0.0])*(a-distance1)/a + np.array([0.58,0.8,1.0])*distance1/a
            elif distance2< b:
                color_map[i]=np.array([255.0/255.0,80.0/255.0,5.0/255.0])*(b-distance2)/b + np.array([0.58,0.8,1.0])*distance2/b
            elif distance3< c:
                color_map[i]=np.array([0.0,0.0,1.0])*(c-distance3)/c + np.array([0.58,0.8,1.0])*distance3/c
            elif distance4< d:
                color_map[i]=np.array([187.0/255.0,0.0,1.0])*(d-distance4)/d + np.array([0.58,0.8,1.0])*distance4/d
            else:
                color_map[i]=np.array([0.58,0.8,1.0]).astype(np.float32)
        return color_map

    xml_head = \
        f"""
    <scene version="0.6.0">
        <integrator type="path">
            <integer name="maxDepth" value="-1"/>
        </integrator>
        <sensor type="perspective">
            <float name="farClip" value="100"/>
            <float name="nearClip" value="0.1"/>
            <transform name="toWorld">
                <lookat origin="{camera_extrinsic['location']}" target="{camera_extrinsic['look_at']}" up="{camera_extrinsic['up']}"/>
            </transform>
            <float name="fov" value="{camera_intrinsic['fov']}"/>
            
            <sampler type="ldsampler">
                <integer name="sampleCount" value="256"/>
            </sampler>
            <film type="hdrfilm">
                <integer name="width" value="{image_size['xres']}"/>
                <integer name="height" value="{image_size['yres']}"/>
                <rfilter type="gaussian"/>
                <boolean name="banner" value="false"/>
            </film>
        </sensor>
        
        <bsdf type="roughplastic" id="surfaceMaterial">
            <string name="distribution" value="ggx"/>
            <float name="alpha" value="0.05"/>
            <float name="intIOR" value="1.46"/>
            <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
        </bsdf>
    """

    xml_ball_segment = \
        """
        <shape type="sphere">
            <float name="radius" value="0.01"/> 
            <transform name="toWorld">
                <translate x="{}" y="{}" z="{}"/>
            </transform>
            <bsdf type="diffuse">
                <rgb name="reflectance" value="{},{},{}"/>
            </bsdf>
        </shape>
    """

    xml_select_ball_segment = \
    """
    <shape type="sphere">
        <float name="radius" value="0.010"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="1,0,0"/>
        </bsdf>
    </shape>
    """


    xml_tail = \
        f"""
        <shape type="rectangle">
            <ref name="bsdf" id="surfaceMaterial"/>
            <transform name="toWorld">
                <scale x="100" y="100" z="1"/>
                <translate x="0" y="0" z="-4"/>
            </transform>
        </shape>
        
        <shape type="rectangle">
            <transform name="toWorld">
                <scale x="10" y="10" z="1"/>
                <lookat origin="{light_extrinsic['location']}" target="{light_extrinsic['look_at']}" up="{light_extrinsic['up']}"/>
            </transform>
            <emitter type="area">
                <rgb name="radiance" value="3,3,3"/>
            </emitter>
        </shape>
    </scene>
    """

    pcl1, feature1, center1, scale1 = standardize_bbox(pts1, feature1, 7000,
                                                        select_point1, select_point_feature1, 
                                                        select_point2, select_point_feature2,
                                                        select_point3, select_point_feature3,
                                                        select_point4, select_point_feature4)
    xml_segments1 = [xml_head]
    color_map1 = colormap(feature1, pcl1, 
                          pcl1[0], select_point_feature1, 
                          pcl1[1], select_point_feature2,
                          pcl1[2], select_point_feature3,
                          pcl1[3], select_point_feature4)
    for i in range(pcl1.shape[0]):
        color = color_map1[i]
        xml_segments1.append(xml_ball_segment.format(
            pcl1[i, 0], pcl1[i, 1], pcl1[i, 2], color[0], color[1], color[2]))
    # xml_segments1.append(xml_select_ball_segment.format(
    #     pcl1[0, 0],pcl1[0, 1],pcl1[0, 2]))
    xml_segments1.append(xml_tail)
    xml_content1 = str.join('', xml_segments1)
    xml_name1 = xml_name1+'.xml'
    xml_path = os.path.join(path, xml_name1)
    with open(xml_path, 'w') as f:
        f.write(xml_content1)

def generate_one_point_heatmap_xml_for_mitsuba(
        pts1,  # [num_points,3]
        feature1,  # [num_points,512]
        select_point1, select_point_feature1,
        image_size={'xres': 640, 'yres': 640},
        camera_intrinsic={'fov': 20},
        camera_extrinsic={'location': "3,3,3",
                          'look_at': "0,0,0", 'up': "0,0,1"},
        light_extrinsic={'location': "-4,4,20",
                         'look_at': "0,0,0", 'up': "0,0,1"},
        path='/your/xml/save/path',
        xml_name1='one'): 
    def standardize_bbox(pcl, feature, points_per_object,
                        select_point1, select_point_feature1):
        if pcl.shape[0] > points_per_object:
            pt_indices = np.random.choice(
                pcl.shape[0], points_per_object, replace=False)
            np.random.shuffle(pt_indices)
            pcl = pcl[pt_indices]  # n by 3

        mins = np.amin(pcl, axis=0)
        maxs = np.amax(pcl, axis=0)
        center = (mins + maxs) / 2.
        scale = np.amax(maxs-mins)
        print("Center: {}, Scale: {}".format(center, scale))
        result = ((pcl - center)/scale).astype(np.float32)  # [-0.5, 0.5]
        result[0]=((select_point1-center)/scale).astype(np.float32)
        return result, feature, center, scale

    # feature[num_point,512]  point_feature[512]
    def colormap(feature, pts,
                select_point1, select_point_feature1):
        color_map=np.zeros((pts.shape[0],3),dtype=np.float32)
        limit1 = np.random.uniform(ra1,rb2)
        for i in range(pts.shape[0]):
            distance1 = np.clip(np.linalg.norm(pts[i] - select_point1)+generate_random_number()/180,0,1)
            a=np.random.uniform(rc3, limit1)
            if distance1< a:
                color_map[i]=np.array([1.0,0.0,0.0])*(a-distance1)/a + np.array([0.58,0.8,1.0])*distance1/a
            else:
                color_map[i]=np.array([0.58,0.8,1.0]).astype(np.float32)
        return color_map

    xml_head = \
        f"""
    <scene version="0.6.0">
        <integrator type="path">
            <integer name="maxDepth" value="-1"/>
        </integrator>
        <sensor type="perspective">
            <float name="farClip" value="100"/>
            <float name="nearClip" value="0.1"/>
            <transform name="toWorld">
                <lookat origin="{camera_extrinsic['location']}" target="{camera_extrinsic['look_at']}" up="{camera_extrinsic['up']}"/>
            </transform>
            <float name="fov" value="{camera_intrinsic['fov']}"/>
            
            <sampler type="ldsampler">
                <integer name="sampleCount" value="256"/>
            </sampler>
            <film type="hdrfilm">
                <integer name="width" value="{image_size['xres']}"/>
                <integer name="height" value="{image_size['yres']}"/>
                <rfilter type="gaussian"/>
                <boolean name="banner" value="false"/>
            </film>
        </sensor>
        
        <bsdf type="roughplastic" id="surfaceMaterial">
            <string name="distribution" value="ggx"/>
            <float name="alpha" value="0.05"/>
            <float name="intIOR" value="1.46"/>
            <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
        </bsdf>
    """

    xml_ball_segment = \
        """
        <shape type="sphere">
            <float name="radius" value="0.01"/> 
            <transform name="toWorld">
                <translate x="{}" y="{}" z="{}"/>
            </transform>
            <bsdf type="diffuse">
                <rgb name="reflectance" value="{},{},{}"/>
            </bsdf>
        </shape>
    """

    xml_select_ball_segment = \
    """
    <shape type="sphere">
        <float name="radius" value="0.010"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="1,0,0"/>
        </bsdf>
    </shape>
    """


    xml_tail = \
        f"""
        <shape type="rectangle">
            <ref name="bsdf" id="surfaceMaterial"/>
            <transform name="toWorld">
                <scale x="100" y="100" z="1"/>
                <translate x="0" y="0" z="-4"/>
            </transform>
        </shape>
        
        <shape type="rectangle">
            <transform name="toWorld">
                <scale x="10" y="10" z="1"/>
                <lookat origin="{light_extrinsic['location']}" target="{light_extrinsic['look_at']}" up="{light_extrinsic['up']}"/>
            </transform>
            <emitter type="area">
                <rgb name="radiance" value="3,3,3"/>
            </emitter>
        </shape>
    </scene>
    """

    pcl1, feature1, center1, scale1 = standardize_bbox(pts1, feature1, 7000,
                                                        select_point1, select_point_feature1)
    xml_segments1 = [xml_head]
    color_map1 = colormap(feature1, pcl1, 
                          pcl1[0], select_point_feature1)
    for i in range(pcl1.shape[0]):
        color = color_map1[i]
        xml_segments1.append(xml_ball_segment.format(
            pcl1[i, 0], pcl1[i, 1], pcl1[i, 2], color[0], color[1], color[2]))
    # xml_segments1.append(xml_select_ball_segment.format(
    #     pcl1[0, 0],pcl1[0, 1],pcl1[0, 2]))
    xml_segments1.append(xml_tail)
    xml_content1 = str.join('', xml_segments1)
    xml_name1 = xml_name1+'.xml'
    xml_path = os.path.join(path, xml_name1)
    with open(xml_path, 'w') as f:
        f.write(xml_content1)

def generate_two_grasp_heatmap_xml_for_mitsuba(
        pts1,  # [num_points,3]
        feature1,  # [num_points,512]
        select_point1, select_point_feature1,
        select_point2, select_point_feature2,
        image_size={'xres': 640, 'yres': 640},
        camera_intrinsic={'fov': 20},
        camera_extrinsic={'location': "3,3,3",
                          'look_at': "0,0,0", 'up': "0,0,1"},
        light_extrinsic={'location': "-4,4,20",
                         'look_at': "0,0,0", 'up': "0,0,1"},
        path='/your/xml/save/path',
        xml_name1='one'): 
    def standardize_bbox(pcl, feature, points_per_object,
                        select_point1, select_point_feature1, 
                        select_point2, select_point_feature2):
        if pcl.shape[0] > points_per_object:
            pt_indices = np.random.choice(
                pcl.shape[0], points_per_object, replace=False)
            np.random.shuffle(pt_indices)
            pcl = pcl[pt_indices]  # n by 3

        mins = np.amin(pcl, axis=0)
        maxs = np.amax(pcl, axis=0)
        center = (mins + maxs) / 2.
        scale = np.amax(maxs-mins)
        print("Center: {}, Scale: {}".format(center, scale))
        result = ((pcl - center)/scale).astype(np.float32)  # [-0.5, 0.5]
        result[0]=((select_point1-center)/scale).astype(np.float32)
        result[1]=((select_point2-center)/scale).astype(np.float32)
        return result, feature, center, scale

    # feature[num_point,512]  point_feature[512]
    def colormap(feature, pts,
                select_point1, select_point_feature1, 
                select_point2, select_point_feature2):
        color_map=np.zeros((pts.shape[0],3),dtype=np.float32)
        limit1 = np.random.uniform(ra1,rb2)
        limit2 = np.random.uniform(ra1,rb2)
        for i in range(pts.shape[0]):
            distance1 = np.clip(np.linalg.norm(pts[i] - select_point1)+generate_random_number()/180,0,1)
            distance2 = np.clip(np.linalg.norm(pts[i] - select_point2)+generate_random_number()/180,0,1)
            a=np.random.uniform(rc3, limit1)
            b=np.random.uniform(rc3, limit2)
            if distance1< a:
                color_map[i]=np.array([1.0,0.0,0.0])*(a-distance1)/a + np.array([0.58,0.8,1.0])*distance1/a
            elif distance2< b:
                color_map[i]=np.array([255.0/255.0,80.0/255.0,5.0/255.0])*(b-distance2)/b + np.array([0.58,0.8,1.0])*distance2/b
            else:
                color_map[i]=np.array([0.58,0.8,1.0]).astype(np.float32)
        return color_map

    xml_head = \
        f"""
    <scene version="0.6.0">
        <integrator type="path">
            <integer name="maxDepth" value="-1"/>
        </integrator>
        <sensor type="perspective">
            <float name="farClip" value="100"/>
            <float name="nearClip" value="0.1"/>
            <transform name="toWorld">
                <lookat origin="{camera_extrinsic['location']}" target="{camera_extrinsic['look_at']}" up="{camera_extrinsic['up']}"/>
            </transform>
            <float name="fov" value="{camera_intrinsic['fov']}"/>
            
            <sampler type="ldsampler">
                <integer name="sampleCount" value="256"/>
            </sampler>
            <film type="hdrfilm">
                <integer name="width" value="{image_size['xres']}"/>
                <integer name="height" value="{image_size['yres']}"/>
                <rfilter type="gaussian"/>
                <boolean name="banner" value="false"/>
            </film>
        </sensor>
        
        <bsdf type="roughplastic" id="surfaceMaterial">
            <string name="distribution" value="ggx"/>
            <float name="alpha" value="0.05"/>
            <float name="intIOR" value="1.46"/>
            <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
        </bsdf>
    """

    xml_ball_segment = \
        """
        <shape type="sphere">
            <float name="radius" value="0.01"/> 
            <transform name="toWorld">
                <translate x="{}" y="{}" z="{}"/>
            </transform>
            <bsdf type="diffuse">
                <rgb name="reflectance" value="{},{},{}"/>
            </bsdf>
        </shape>
    """

    xml_select_ball_segment = \
    """
    <shape type="sphere">
        <float name="radius" value="0.010"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="1,0,0"/>
        </bsdf>
    </shape>
    """


    xml_tail = \
        f"""
        <shape type="rectangle">
            <ref name="bsdf" id="surfaceMaterial"/>
            <transform name="toWorld">
                <scale x="100" y="100" z="1"/>
                <translate x="0" y="0" z="-4"/>
            </transform>
        </shape>
        
        <shape type="rectangle">
            <transform name="toWorld">
                <scale x="10" y="10" z="1"/>
                <lookat origin="{light_extrinsic['location']}" target="{light_extrinsic['look_at']}" up="{light_extrinsic['up']}"/>
            </transform>
            <emitter type="area">
                <rgb name="radiance" value="3,3,3"/>
            </emitter>
        </shape>
    </scene>
    """

    pcl1, feature1, center1, scale1 = standardize_bbox(pts1, feature1, 7000,
                                                        select_point1, select_point_feature1, 
                                                        select_point2, select_point_feature2)
    xml_segments1 = [xml_head]
    color_map1 = colormap(feature1, pcl1, 
                          pcl1[0], select_point_feature1, 
                          pcl1[1], select_point_feature2)
    for i in range(pcl1.shape[0]):
        color = color_map1[i]
        xml_segments1.append(xml_ball_segment.format(
            pcl1[i, 0], pcl1[i, 1], pcl1[i, 2], color[0], color[1], color[2]))
    # xml_segments1.append(xml_select_ball_segment.format(
    #     pcl1[0, 0],pcl1[0, 1],pcl1[0, 2]))
    xml_segments1.append(xml_tail)
    xml_content1 = str.join('', xml_segments1)
    xml_name1 = xml_name1+'.xml'
    xml_path = os.path.join(path, xml_name1)
    with open(xml_path, 'w') as f:
        f.write(xml_content1)


def generate_six_points_heatmap_xml_for_mitsuba(
        pts1,  # [num_points,3]
        feature1,  # [num_points,512]
        select_point1, select_point_feature1,
        select_point2, select_point_feature2,
        select_point3, select_point_feature3,
        select_point4, select_point_feature4,
        select_point5, select_point_feature5,
        select_point6, select_point_feature6,
        image_size={'xres': 640, 'yres': 640},
        camera_intrinsic={'fov': 20},
        camera_extrinsic={'location': "3,3,3",
                          'look_at': "0,0,0", 'up': "0,0,1"},
        light_extrinsic={'location': "-4,4,20",
                         'look_at': "0,0,0", 'up': "0,0,1"},
        path='/your/xml/save/path',
        xml_name1='one'): 
    def standardize_bbox(pcl, feature, points_per_object,
                        select_point1, select_point_feature1, 
                        select_point2, select_point_feature2,
                        select_point3, select_point_feature3,
                        select_point4, select_point_feature4,
                        select_point5, select_point_feature5,
                        select_point6, select_point_feature6):
        if pcl.shape[0] > points_per_object:
            pt_indices = np.random.choice(
                pcl.shape[0], points_per_object, replace=False)
            np.random.shuffle(pt_indices)
            pcl = pcl[pt_indices]  # n by 3

        mins = np.amin(pcl, axis=0)
        maxs = np.amax(pcl, axis=0)
        center = (mins + maxs) / 2.
        scale = np.amax(maxs-mins)
        print("Center: {}, Scale: {}".format(center, scale))
        result = ((pcl - center)/scale).astype(np.float32)  # [-0.5, 0.5]
        result[0]=((select_point1-center)/scale).astype(np.float32)
        result[1]=((select_point2-center)/scale).astype(np.float32)
        result[2]=((select_point3-center)/scale).astype(np.float32)
        result[3]=((select_point4-center)/scale).astype(np.float32)
        result[4]=((select_point5-center)/scale).astype(np.float32)
        result[5]=((select_point6-center)/scale).astype(np.float32)
        return result, feature, center, scale
    def colormap(pts,
            select_point1,
            select_point2,
            select_point3,
            select_point4,
            select_point5,
            select_point6):
        color_map=np.zeros((pts.shape[0],3),dtype=np.float32)
        limit1 = np.random.uniform(0.1,0.16)
        limit2 = np.random.uniform(0.1,0.16)
        limit3 = np.random.uniform(0.1,0.16)
        limit4 = np.random.uniform(0.1,0.16)
        limit5 = np.random.uniform(0.1,0.16)
        limit6 = np.random.uniform(0.1,0.16)
        for i in range(pts.shape[0]):
            distance1 = np.clip(np.linalg.norm(pts[i] - select_point1)+generate_random_number()/180,0,1)
            distance2 = np.clip(np.linalg.norm(pts[i] - select_point2)+generate_random_number()/180,0,1)
            distance3 = np.clip(np.linalg.norm(pts[i] - select_point3)+generate_random_number()/180,0,1)
            distance4 = np.clip(np.linalg.norm(pts[i] - select_point4)+generate_random_number()/180,0,1)
            distance5 = np.clip(np.linalg.norm(pts[i] - select_point5)+generate_random_number()/180,0,1)
            distance6 = np.clip(np.linalg.norm(pts[i] - select_point6)+generate_random_number()/180,0,1)
            a=np.random.uniform(0.06, limit1)
            b=np.random.uniform(0.06, limit2)
            c=np.random.uniform(0.06, limit3)
            d=np.random.uniform(0.06, limit4)
            e=np.random.uniform(0.06, limit5)
            f=np.random.uniform(0.06, limit6)
            if distance1< a:
                color_map[i]=np.array([1.0,0.0,0.0])*(a-distance1)/a + np.array([0.58,0.8,1.0])*distance1/a
            elif distance2< b:
                color_map[i]=np.array([255.0/255.0,80.0/255.0,5.0/255.0])*(b-distance2)/b + np.array([0.58,0.8,1.0])*distance2/b
            elif distance3< c:
                color_map[i]=np.array([0.0,0.0,1.0])*(c-distance3)/c + np.array([0.58,0.8,1.0])*distance3/c
            elif distance4< d:
                color_map[i]=np.array([187.0/255.0,0.0,1.0])*(d-distance4)/d + np.array([0.58,0.8,1.0])*distance4/d
            elif distance5< e:
                color_map[i]=np.array([0.0,1.0,0.0])*(e-distance5)/e + np.array([0.58,0.8,1.0])*distance5/e
            elif distance6< f:
                color_map[i]=np.array([1.0,1.0,0.0])*(f-distance6)/f + np.array([0.58,0.8,1.0])*distance6/f
            else:
                color_map[i]=np.array([0.58,0.8,1.0]).astype(np.float32)
        return color_map
    
    xml_head = \
        f"""
    <scene version="0.6.0">
        <integrator type="path">
            <integer name="maxDepth" value="-1"/>
        </integrator>
        <sensor type="perspective">
            <float name="farClip" value="100"/>
            <float name="nearClip" value="0.1"/>
            <transform name="toWorld">
                <lookat origin="{camera_extrinsic['location']}" target="{camera_extrinsic['look_at']}" up="{camera_extrinsic['up']}"/>
            </transform>
            <float name="fov" value="{camera_intrinsic['fov']}"/>
            
            <sampler type="ldsampler">
                <integer name="sampleCount" value="256"/>
            </sampler>
            <film type="hdrfilm">
                <integer name="width" value="{image_size['xres']}"/>
                <integer name="height" value="{image_size['yres']}"/>
                <rfilter type="gaussian"/>
                <boolean name="banner" value="false"/>
            </film>
        </sensor>
        
        <bsdf type="roughplastic" id="surfaceMaterial">
            <string name="distribution" value="ggx"/>
            <float name="alpha" value="0.05"/>
            <float name="intIOR" value="1.46"/>
            <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
        </bsdf>
    """

    xml_ball_segment = \
        """
        <shape type="sphere">
            <float name="radius" value="0.01"/> 
            <transform name="toWorld">
                <translate x="{}" y="{}" z="{}"/>
            </transform>
            <bsdf type="diffuse">
                <rgb name="reflectance" value="{},{},{}"/>
            </bsdf>
        </shape>
    """

    xml_select_ball_segment = \
    """
    <shape type="sphere">
        <float name="radius" value="0.010"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="1,0,0"/>
        </bsdf>
    </shape>
    """


    xml_tail = \
        f"""
        <shape type="rectangle">
            <ref name="bsdf" id="surfaceMaterial"/>
            <transform name="toWorld">
                <scale x="100" y="100" z="1"/>
                <translate x="0" y="0" z="-4"/>
            </transform>
        </shape>
        
        <shape type="rectangle">
            <transform name="toWorld">
                <scale x="10" y="10" z="1"/>
                <lookat origin="{light_extrinsic['location']}" target="{light_extrinsic['look_at']}" up="{light_extrinsic['up']}"/>
            </transform>
            <emitter type="area">
                <rgb name="radiance" value="3,3,3"/>
            </emitter>
        </shape>
    </scene>
    """

    pcl1, feature1, center1, scale1 = standardize_bbox(pts1, feature1, 7000,
                                                        select_point1, select_point_feature1, 
                                                        select_point2, select_point_feature2,
                                                        select_point3, select_point_feature3,
                                                        select_point4, select_point_feature4,
                                                        select_point5, select_point_feature5,
                                                        select_point6, select_point_feature6)
    xml_segments1 = [xml_head]
    color_map1 = colormap(pcl1, pcl1[0], pcl1[1],pcl1[2],pcl1[3],pcl1[4],pcl1[5])
    for i in range(pcl1.shape[0]):
        color = color_map1[i]
        xml_segments1.append(xml_ball_segment.format(
            pcl1[i, 0], pcl1[i, 1], pcl1[i, 2], color[0], color[1], color[2]))
    # xml_segments1.append(xml_select_ball_segment.format(
    #     pcl1[0, 0],pcl1[0, 1],pcl1[0, 2]))
    xml_segments1.append(xml_tail)
    xml_content1 = str.join('', xml_segments1)
    xml_name1 = xml_name1+'.xml'
    xml_path = os.path.join(path, xml_name1)
    with open(xml_path, 'w') as f:
        f.write(xml_content1)




def generate_grasp_place_heatmap_xml_for_mitsuba(
        pts1,  # [num_points,3]
        feature1,  # [num_points,512]
        select_point1, select_point_feature1,
        select_point3, select_point_feature3,
        image_size={'xres': 640, 'yres': 640},
        camera_intrinsic={'fov': 20},
        camera_extrinsic={'location': "3,3,3",
                          'look_at': "0,0,0", 'up': "0,0,1"},
        light_extrinsic={'location': "-4,4,20",
                         'look_at': "0,0,0", 'up': "0,0,1"},
        path='/your/xml/save/path',
        xml_name1='one'): 
    def standardize_bbox(pcl, feature, points_per_object,
                        select_point1, select_point_feature1, 
                        select_point3, select_point_feature3):
        if pcl.shape[0] > points_per_object:
            pt_indices = np.random.choice(
                pcl.shape[0], points_per_object, replace=False)
            np.random.shuffle(pt_indices)
            pcl = pcl[pt_indices]  # n by 3

        mins = np.amin(pcl, axis=0)
        maxs = np.amax(pcl, axis=0)
        center = (mins + maxs) / 2.
        scale = np.amax(maxs-mins)
        print("Center: {}, Scale: {}".format(center, scale))
        result = ((pcl - center)/scale).astype(np.float32)  # [-0.5, 0.5]
        result[0]=((select_point1-center)/scale).astype(np.float32)
        result[2]=((select_point3-center)/scale).astype(np.float32)
        return result, feature, center, scale

    # feature[num_point,512]  point_feature[512]
    def colormap(feature, pts,
                select_point1, select_point_feature1, 
                select_point3, select_point_feature3):
        color_map=np.zeros((pts.shape[0],3),dtype=np.float32)
        limit1 = np.random.uniform(ra1,rb2)
        limit3 = np.random.uniform(ra1,rb2)
        for i in range(pts.shape[0]):
            distance1 = np.clip(np.linalg.norm(pts[i] - select_point1)+generate_random_number()/180,0,1)
            distance3 = np.clip(np.linalg.norm(pts[i] - select_point3)+generate_random_number()/130,0,1)
            a=np.random.uniform(rc3, limit1)
            c=np.random.uniform(rc3, limit3)
            if distance1< a:
                color_map[i]=np.array([1.0,0.0,0.0])*(a-distance1)/a + np.array([0.58,0.8,1.0])*distance1/a
            elif distance3< c:
                color_map[i]=np.array([0.0,0.0,1.0])*(c-distance3)/c + np.array([0.58,0.8,1.0])*distance3/c
            else:
                color_map[i]=np.array([0.58,0.8,1.0]).astype(np.float32)
        return color_map

    xml_head = \
        f"""
    <scene version="0.6.0">
        <integrator type="path">
            <integer name="maxDepth" value="-1"/>
        </integrator>
        <sensor type="perspective">
            <float name="farClip" value="100"/>
            <float name="nearClip" value="0.1"/>
            <transform name="toWorld">
                <lookat origin="{camera_extrinsic['location']}" target="{camera_extrinsic['look_at']}" up="{camera_extrinsic['up']}"/>
            </transform>
            <float name="fov" value="{camera_intrinsic['fov']}"/>
            
            <sampler type="ldsampler">
                <integer name="sampleCount" value="256"/>
            </sampler>
            <film type="hdrfilm">
                <integer name="width" value="{image_size['xres']}"/>
                <integer name="height" value="{image_size['yres']}"/>
                <rfilter type="gaussian"/>
                <boolean name="banner" value="false"/>
            </film>
        </sensor>
        
        <bsdf type="roughplastic" id="surfaceMaterial">
            <string name="distribution" value="ggx"/>
            <float name="alpha" value="0.05"/>
            <float name="intIOR" value="1.46"/>
            <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
        </bsdf>
    """

    xml_ball_segment = \
        """
        <shape type="sphere">
            <float name="radius" value="0.01"/> 
            <transform name="toWorld">
                <translate x="{}" y="{}" z="{}"/>
            </transform>
            <bsdf type="diffuse">
                <rgb name="reflectance" value="{},{},{}"/>
            </bsdf>
        </shape>
    """

    xml_select_ball_segment = \
    """
    <shape type="sphere">
        <float name="radius" value="0.010"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="1,0,0"/>
        </bsdf>
    </shape>
    """


    xml_tail = \
        f"""
        <shape type="rectangle">
            <ref name="bsdf" id="surfaceMaterial"/>
            <transform name="toWorld">
                <scale x="100" y="100" z="1"/>
                <translate x="0" y="0" z="-4"/>
            </transform>
        </shape>
        
        <shape type="rectangle">
            <transform name="toWorld">
                <scale x="10" y="10" z="1"/>
                <lookat origin="{light_extrinsic['location']}" target="{light_extrinsic['look_at']}" up="{light_extrinsic['up']}"/>
            </transform>
            <emitter type="area">
                <rgb name="radiance" value="3,3,3"/>
            </emitter>
        </shape>
    </scene>
    """

    pcl1, feature1, center1, scale1 = standardize_bbox(pts1, feature1, 7000,
                                                        select_point1, select_point_feature1, 
                                                        select_point3, select_point_feature3)
    xml_segments1 = [xml_head]
    color_map1 = colormap(feature1, pcl1, 
                          pcl1[0], select_point_feature1, 
                          pcl1[2], select_point_feature3)
    for i in range(pcl1.shape[0]):
        color = color_map1[i]
        xml_segments1.append(xml_ball_segment.format(
            pcl1[i, 0], pcl1[i, 1], pcl1[i, 2], color[0], color[1], color[2]))
    # xml_segments1.append(xml_select_ball_segment.format(
    #     pcl1[0, 0],pcl1[0, 1],pcl1[0, 2]))
    xml_segments1.append(xml_tail)
    xml_content1 = str.join('', xml_segments1)
    xml_name1 = xml_name1+'.xml'
    xml_path = os.path.join(path, xml_name1)
    with open(xml_path, 'w') as f:
        f.write(xml_content1)


def turn_xml_to_png(path='/your/save/path',
                    xml_name='haha',
                    image_name='hahaha'):
    mi.set_variant("scalar_rgb")
    xml_name = xml_name+'.xml'
    png_name = image_name+'.png'
    scene_path = os.path.join(path, xml_name)
    scene = mi.load_file(scene_path)
    image = mi.render(scene)
    png_path = os.path.join(path, png_name)
    mi.util.write_bitmap(png_path, image)


def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True


def get_four_points_heatmap(pts1,
                select_point1, select_point_feature1,
                select_point2, select_point_feature2,
                select_point3, select_point_feature3,
                select_point4, select_point_feature4,
                save_path='/home/luhr/correspondence/softgym_cloth/visualize/mitsuba_heatmap',
                name='test'):
    exists_or_mkdir(save_path)
    pts1[:, 2] *= 2.0
    image_size = {'xres': 640, 'yres': 640}
    camera_extrinsic = {'location': f"{0},{0},{4}",
                        'look_at': "0,0,0", 'up': "0,1,0"}
    light_extrinsic = {'location': "6,-6,16",
                       'look_at': "0,0,0", 'up': "0,0,1"}
    generate_four_points_heatmap_xml_for_mitsuba(pts1[:,:3],pts1[:,-512:],
                                     select_point1, select_point_feature1,
                                     select_point2, select_point_feature2,
                                     select_point3, select_point_feature3,
                                     select_point4, select_point_feature4,
                                     image_size=image_size, 
                                     camera_extrinsic=camera_extrinsic,
                                     light_extrinsic=light_extrinsic, 
                                     path=save_path, 
                                     xml_name1=name)
    turn_xml_to_png(save_path, xml_name=name, image_name=name)

def get_one_point_heatmap(pts1,
                select_point1, select_point_feature1,
                save_path='/home/luhr/correspondence/softgym_cloth/visualize/mitsuba_heatmap',
                name='test'):
    exists_or_mkdir(save_path)
    pts1[:, 2] *= 2.0
    image_size = {'xres': 640, 'yres': 640}
    camera_extrinsic = {'location': f"{0},{0},{4}",
                        'look_at': "0,0,0", 'up': "0,1,0"}
    light_extrinsic = {'location': "6,-6,16",
                       'look_at': "0,0,0", 'up': "0,0,1"}
    generate_one_point_heatmap_xml_for_mitsuba(pts1[:,:3],pts1[:,-512:],
                                     select_point1, select_point_feature1,
                                     image_size=image_size, 
                                     camera_extrinsic=camera_extrinsic,
                                     light_extrinsic=light_extrinsic, 
                                     path=save_path, 
                                     xml_name1=name)
    turn_xml_to_png(save_path, xml_name=name, image_name=name)

def get_two_grasp_heatmap(pts1,
                select_point1, select_point_feature1,
                select_point2, select_point_feature2,
                save_path='/home/luhr/correspondence/softgym_cloth/visualize/mitsuba_heatmap',
                name='test'):
    exists_or_mkdir(save_path)
    pts1[:, 2] *= 2.0  #2
    image_size = {'xres': 640, 'yres': 640}
    camera_extrinsic = {'location': f"{0},{0},{4}",
                        'look_at': "0,0,0", 'up': "0,1,0"}
    light_extrinsic = {'location': "6,-6,16",
                       'look_at': "0,0,0", 'up': "0,0,1"}
    generate_two_grasp_heatmap_xml_for_mitsuba(pts1[:,:3],pts1[:,-512:],
                                     select_point1, select_point_feature1,
                                     select_point2, select_point_feature2,
                                     image_size=image_size, 
                                     camera_extrinsic=camera_extrinsic,
                                     light_extrinsic=light_extrinsic, 
                                     path=save_path, 
                                     xml_name1=name)
    turn_xml_to_png(save_path, xml_name=name, image_name=name)

def get_grasp_place_heatmap(pts1,
                select_point1, select_point_feature1,
                select_point2, select_point_feature2,
                save_path='/home/luhr/correspondence/softgym_cloth/visualize/mitsuba_heatmap',
                name='test'):
    exists_or_mkdir(save_path)
    pts1[:, 2] *= 2.0
    image_size = {'xres': 640, 'yres': 640}
    camera_extrinsic = {'location': f"{0},{0},{4}",
                        'look_at': "0,0,0", 'up': "0,1,0"}
    light_extrinsic = {'location': "6,-6,16",
                       'look_at': "0,0,0", 'up': "0,0,1"}
    generate_grasp_place_heatmap_xml_for_mitsuba(pts1[:,:3],pts1[:,-512:],
                                     select_point1, select_point_feature1,
                                     select_point2, select_point_feature2,
                                     image_size=image_size, 
                                     camera_extrinsic=camera_extrinsic,
                                     light_extrinsic=light_extrinsic, 
                                     path=save_path, 
                                     xml_name1=name)
    turn_xml_to_png(save_path, xml_name=name, image_name=name)


def get_six_points_heatmap(pts1,
                select_point1, select_point_feature1,
                select_point2, select_point_feature2,
                select_point3, select_point_feature3,
                select_point4, select_point_feature4,
                select_point5, select_point_feature5,
                select_point6, select_point_feature6,
                save_path='/home/luhr/correspondence/softgym_cloth/visualize/mitsuba_heatmap',
                name='test-six'):
    exists_or_mkdir(save_path)
    pts1[:, 2] *= 2.0
    image_size = {'xres': 640, 'yres': 640}
    camera_extrinsic = {'location': f"{0},{0},{4}",
                        'look_at': "0,0,0", 'up': "0,1,0"}
    light_extrinsic = {'location': "6,-6,16",
                       'look_at': "0,0,0", 'up': "0,0,1"}
    generate_six_points_heatmap_xml_for_mitsuba(pts1[:,:3],pts1[:,-512:],
                                     select_point1, select_point_feature1,
                                     select_point2, select_point_feature2,
                                     select_point3, select_point_feature3,
                                     select_point4, select_point_feature4,
                                     select_point5, select_point_feature5,
                                     select_point6, select_point_feature6,
                                     image_size=image_size, 
                                     camera_extrinsic=camera_extrinsic,
                                     light_extrinsic=light_extrinsic, 
                                     path=save_path, 
                                     xml_name1=name)
    turn_xml_to_png(save_path, xml_name=name, image_name=name)



# save_path=r"D:\code_field\gitgitgit\PointFlowRenderer\heatmapresultfour\points7000"
# pts1=np.load(r"D:\code_field\gitgitgit\PointFlowRenderer\visual_demo\all_new\00037\0.npy")
# pts2=np.load(r"D:\code_field\gitgitgit\PointFlowRenderer\visual_demo\all_new\00557\0.npy")
# select_point1 = pts1[0,:3]
# select_point_feature1 = pts1[0,-512:]
# select_point2 = pts1[0,:3]
# select_point_feature2 = pts1[0,-512:]
# select_point3 = pts1[0,:3]
# select_point_feature3 = pts1[0,-512:]
# select_point4 = pts1[0,:3]
# select_point_feature4 = pts1[0,-512:]
# for i in range(pts1.shape[0]):
#     if (select_point1[0]+select_point1[1])>(pts1[i,0]+pts1[i,1]): # 选最靠左上的点
#         select_point1 = pts1[i,:3]
#         select_point_feature1 = pts1[i,-512:]
#     if (-select_point2[0]+select_point2[1])>(-pts1[i,0]+pts1[i,1]): # 选最靠右上的点
#         select_point2 = pts1[i,:3]
#         select_point_feature2 = pts1[i,-512:]
#     if (select_point3[0]-select_point3[1])>(pts1[i,0]-pts1[i,1]): # 选最靠左下的点
#         select_point3 = pts1[i,:3]
#         select_point_feature3 = pts1[i,-512:]
#     if (-select_point4[0]-select_point4[1])>(-pts1[i,0]-pts1[i,1]): # 选最靠右下的点
#         select_point4 = pts1[i,:3]
#         select_point_feature4 = pts1[i,-512:]

# get_grasp_place_heatmap(pts1,
#             select_point1, select_point_feature1,
#             select_point2, select_point_feature2,
#             # select_point3, select_point_feature3,
#             # select_point4, select_point_feature4,
#             save_path,
#             'Testgraspplace')