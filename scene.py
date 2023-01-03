# manim-community version

from manim import *
from numpy import linalg as LA
# import matplotlib.pyplot as plt
# from os.path import join

class Test(ThreeDScene):

    def construct(self):
        self.build_axes_and_orientation()
        self.build_polyhedron()
        self.set_rotation()
        self.build_labels()

        self.set_axis_after_rotation()
        self.build_axis_loops()
        self.biuld_labels_after_rotation()
        self.build_cone()
        self.build_angle_arcs()

        # 四棱锥淡入
        self.fadeIn(self.pyramid)
        # 四棱锥的坐标系淡入
        self.fadeIn(
            self.axisX_pyramid,
            self.axisY_pyramid,
            self.axisZ_pyramid)
        # 转动轴及标签淡入
        self.fadeIn(self.vector_n,self.label_n)
        # 转动
        self.play(*map(lambda obj:Rotate(obj,self.rotate_angle,\
            self.axis_n,about_point=ORIGIN),
                [self.pyramid,
                self.axisX_pyramid,
                self.axisY_pyramid,
                self.axisZ_pyramid]))
        self.wait()
        # 四棱锥、转动轴及标签淡出
        
        # return
        self.fadeOut(self.pyramid,self.vector_n,self.label_n)
        # 转动后的坐标系标签、圈圈淡入
        self.fadeIn(self.label_x_p,
                    self.label_y_p,
                    self.label_z_p,
                    self.xy_p_loop)
        # 静止坐标系、标签、圈圈淡入
        self.fadeIn(self.axisX,self.axisY,self.axisZ,
                    self.xy_loop,self.xz_loop,
                    self.label_x,self.label_y,self.label_z)
        # x'y'轴、标签、圈圈淡出
        self.fadeOut(
            self.axisX_pyramid,self.axisY_pyramid,
            # self.xy_loop,
            self.xy_p_loop,
            self.label_x_p,self.label_y_p
            )

        # self.build_angle_arcs()
        # 圆弧及标签淡入。此时两个圆弧是重合的。
        self.fadeIn(
            self.arc_beta_1,
            self.arc_beta_1_label,
            self.arc_beta_2,
            wait=False
        )
        # # xz圈圈淡入
        # self.play(FadeIn(self.xz_loop))
        # 圆弧绕z轴转动
        self.play(
            Rotate(self.arc_beta_2,-self.rotate_alpha,Z_AXIS,about_point=ORIGIN)
            )
        # 圆弧标签淡入
        self.play(
            FadeIn(self.arc_beta_2_label),
            FadeIn(self.arc_beta_2_line)
        )
        # self.wait()
        # 圆弧及标签淡出
        self.fadeOut(
            self.arc_beta_1_label,
            self.arc_beta_1,
            self.arc_beta_2_label,
            self.arc_beta_2,
            wait=False
        )
        # 静止的z轴淡入、x轴标签淡出
        self.play(FadeIn(self.axisZ3),FadeOut(self.label_x))
        # x,z轴转动
        self.play(
            Rotate(self.axisZ,self.rotate_beta,Y_AXIS,about_point=ORIGIN),
            Rotate(self.axisX,self.rotate_beta,Y_AXIS,about_point=ORIGIN)
        )
        # 转动后的x,z标签淡入，辅助线和圈圈淡出
        self.play(FadeIn(self.label_x3),
                  FadeIn(self.label_z3),
                  FadeOut(self.arc_beta_2_line),
                #   FadeOut(self.xz_loop)
                  )
        # 圆锥面淡入
        # self.build_cone()
        self.wait()
        self.fadeIn(self.cone)
        # self.fadeOut(self.cone)
        # alpha角度及标签淡入
        self.fadeIn(self.alpha_1_line,
                    self.alpha_2_line,
                    self.arc_alpha_label)
        # alpha角度及标签淡出
        self.fadeOut(self.alpha_1_line,
                    self.alpha_2_line,
                    self.arc_alpha_label,
        #             wait=False)
        # # z'x'标签淡出
        # self.fadeOut(
                    self.label_z3,
                    self.label_x3,
                    self.label_y,
                    # self.label_z_p
                    wait=False
                    )
        # 第二次转动
        self.play(
            Rotate(self.axisX,self.rotate_alpha,Z_AXIS,about_point=ORIGIN),
            Rotate(self.axisY,self.rotate_alpha,Z_AXIS,about_point=ORIGIN),
            Rotate(self.axisZ,self.rotate_alpha,Z_AXIS,about_point=ORIGIN),
        )
        self.wait()
        # 标签、圈圈大调整
        # 圈圈、圆锥面、静止的z轴和四棱锥的z轴、z标签淡出
        self.fadeOut(self.xy_loop,
                    self.xz_loop,
                    self.cone,
                    self.axisZ_pyramid,
                    self.axisZ3,
                    self.label_z,
                    wait=False
                    )
        # x2 y2标签淡入
        self.fadeIn(
                    self.label_x4,
                    self.label_y4
        )
        # x' y'轴及标签、圈圈淡入
        self.fadeIn(self.axisX_pyramid,
                    self.axisY_pyramid,
                    self.label_x_p,
                    self.label_y_p,
                    self.xy_p_loop,
                    )
        
        # gamma角度及标签淡入
        self.fadeIn(self.arc_gamma_1,
                    self.arc_gamma_2,
                    self.arc_gamma_1_label,
                    self.arc_gamma_2_label
                    )
        # 淡出
        self.fadeOut(self.arc_gamma_1,
                    self.arc_gamma_2,
                    self.arc_gamma_1_label,
                    self.arc_gamma_2_label,
                    self.label_x4,
                    self.label_y4,
                    wait = False
                    )
        # 第三次转动
        self.play(*map(lambda obj:Rotate(obj,
                                            self.rotate_gamma,
                                            self.z_axis_p_end,
                                            about_point=ORIGIN),
                          [self.axisX,self.axisY])
                    )
        self.fadeOut(
                    # self.xy_p_loop,
                    self.axisX_pyramid,
                    self.axisY_pyramid,
        )
        self.fadeIn(
                    self.axisZ_origin,
                    self.axisX_origin,
                    self.axisY_origin,
                    self.label_z,
                    self.label_y,
                    self.label_x,
                    self.xy_loop            
        )

    def build_cone(self):
        self.cone_height = 1
        self.cone_base_radius = np.tan(self.rotate_beta)*self.cone_height
        self.cone = Cone(self.cone_base_radius,self.cone_height,-Z_AXIS,
            resolution=8,fill_opacity=0.5,checkerboard_colors=None, fill_color=PURPLE,stroke_width=0)

    def build_angle_arcs(self):
        arc_radius = 1.3
        radius_range = [arc_radius-0.02,arc_radius]
        # angle_range = [90*DEGREES-self.rotate_beta,90*DEGREES]
        angle_range = [-self.rotate_beta,0]
        arc_beta_1_normal_axis = np.cross(self.z_axis_p_end,Z_AXIS)
        arc_beta_1_normal_axis = normalize(arc_beta_1_normal_axis)
        self.arc_beta_1 = Sector(radius_range=radius_range,angle_range=angle_range,center_point=ORIGIN,
            normal_axis=arc_beta_1_normal_axis,start_direction=Z_AXIS,
            resolution=8,fill_opacity=1,checkerboard_colors=None, fill_color=ORANGE,stroke_width=0)
        # self.arc_beta_1.rotate(90*DEGREES,X_AXIS,about_point=ORIGIN)
        # self.arc_beta_1.rotate(self.rotate_alpha,Z_AXIS,about_point=ORIGIN)
        self.arc_beta_1_label = MathTex(r"\beta").scale(0.8)
        self.arc_beta_1_label.color = BLACK
        arc_beta_1_label_pos = normalize(Z_AXIS+self.z_axis_p_end)*(arc_radius+0.3)
        self.arc_beta_1_label.move_to(arc_beta_1_label_pos)
        self.arc_beta_1_label.rotate(self.orientation_rotate_angle,\
                self.orientation_rotate_axis)

        self.arc_beta_2 = self.arc_beta_1.copy()
        self.arc_beta_2_label = self.arc_beta_1_label.copy()
        arc_beta_2_label_pos = normalize(np.array([np.sin(self.rotate_beta/2),0,np.cos(self.rotate_beta/2)]))*(arc_radius+0.2)
        self.arc_beta_2_label.move_to(arc_beta_2_label_pos)

        arc_beta_2_line_end = np.array([np.sin(self.rotate_beta),0,np.cos(self.rotate_beta)])*arc_radius*1.5
        self.arc_beta_2_line = Line3D(start=ORIGIN,end=arc_beta_2_line_end,resolution=8,color=GRAY)

        alpha_1_line_start = np.array([0,0,self.cone_height])
        alpha_1_line_end = np.array([self.cone_base_radius,0,self.cone_height])
        self.alpha_1_line = Line3D(start=alpha_1_line_start,
                                   end=alpha_1_line_end,
                                   resolution=8,
                                   color=ORANGE)
        self.alpha_1_line.radius = 0.5
        self.alpha_1_line.fill_opacity = 0.5
        self.alpha_2_line = self.alpha_1_line.copy().rotate(90*DEGREES,Z_AXIS,about_point=ORIGIN)
        self.arc_alpha_label = MathTex(r"\alpha").scale(0.8)
        arc_alpha_label_pos = normalize(np.array([self.cone_base_radius,self.cone_base_radius,0]))*(arc_radius-0.4)+alpha_1_line_start
        self.arc_alpha_label.move_to(arc_alpha_label_pos)
        self.arc_alpha_label.color = BLACK
        self.arc_alpha_label.rotate(self.orientation_rotate_angle,\
                self.orientation_rotate_axis)

        gamma_angle_range = sorted([0,self.rotate_gamma])
        self.arc_gamma_1 = Sector(radius_range=radius_range,angle_range=gamma_angle_range,center_point=ORIGIN,
            normal_axis=self.z_axis_p_end,start_direction=self.axisX4_end,
            resolution=8,fill_opacity=1,checkerboard_colors=None, fill_color=ORANGE,stroke_width=0)
        self.arc_gamma_2 = self.arc_gamma_1.copy().rotate(90*DEGREES,self.z_axis_p_end,about_point=ORIGIN)

        self.arc_gamma_1_label = MathTex(r"\gamma").scale(0.8)
        arc_gamma_1_label_pos = normalize(self.x_axis_p_end+self.axisX4_end)*(arc_radius+0.3)
        self.arc_gamma_1_label.move_to(arc_gamma_1_label_pos)
        self.arc_gamma_1_label.rotate(self.orientation_rotate_angle,\
                self.orientation_rotate_axis)
        self.arc_gamma_1_label.color = BLACK

        self.arc_gamma_2_label = MathTex(r"\gamma").scale(0.8)
        arc_gamma_2_label_pos = normalize(self.y_axis_p_end+self.axisY4_end)*(arc_radius+0.3)
        self.arc_gamma_2_label.move_to(arc_gamma_2_label_pos)
        self.arc_gamma_2_label.rotate(self.orientation_rotate_angle,\
                self.orientation_rotate_axis)
        self.arc_gamma_2_label.color = BLACK
        
        

    def fadeIn(self,*args,wait=True,duration=1):
        self.play(*map(FadeIn,args))
        if wait:
            self.wait(duration)
        
    def fadeOut(self,*args,wait=True,duration=1):
        self.play(*map(FadeOut,args))
        if wait:
            self.wait(duration)

    def biuld_labels_after_rotation(self):
        # 两个坐标系的标签
        ratio = 1.1
        self.label_x_p = MathTex("x'")
        self.label_y_p = MathTex("y'")
        self.label_z_p = MathTex("z'")
        self.label_x_p.move_to(self.x_axis_p_end*(ratio+0.2))
        self.label_y_p.move_to(self.y_axis_p_end*(ratio+0.2))
        self.label_z_p.move_to(self.z_axis_p_end*(ratio+0.2)-self.x_axis_p_end*0.1)

        self.label_x = MathTex("x")
        self.label_y = MathTex("y")
        self.label_z = MathTex("z")
        self.label_x.move_to(X_AXIS*2*(ratio+0.1)+Y_AXIS*0.2)
        self.label_y.move_to(Y_AXIS*2*(ratio))
        self.label_z.move_to(Z_AXIS*2*(ratio+0))

        self.label_x3 = MathTex("x_1")
        self.label_x3.move_to(self.axisX3_end*(ratio+0.2))
        self.label_z3 = MathTex("z_1")
        self.label_z3.move_to(self.axisZ3_end*(ratio+0.2))

        self.label_x4 = MathTex("x_2")
        self.label_x4.move_to(self.axisX4_end*(ratio+0.2))
        self.label_y4 = MathTex("y_2")
        self.label_y4.move_to(self.axisY4_end*(ratio+0.2))

        for label in [
            self.label_x,self.label_y,self.label_z,
            self.label_x_p,self.label_y_p,self.label_z_p,
            self.label_x3,self.label_z3,
            self.label_x4,self.label_y4
        ]:
            label.color = BLACK
            label.rotate(self.orientation_rotate_angle,\
                self.orientation_rotate_axis)

    def build_polyhedron(self):
        vertex_coords = [
            [0,0,1],
            [0,1,0],
            [-1,0,0],
            [0,-1,0],
            [1,0,0],
        ]
        faces_list = [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 1],
            [1, 2, 3, 4]
        ]
        self.pyramid = Polyhedron(vertex_coords, faces_list)

    def build_axes_and_orientation(self):
        self.orientation_phi = 70*DEGREES
        self.orientation_theta = 30*DEGREES

        self.set_camera_orientation(phi=self.orientation_phi, theta=self.orientation_theta)
        self.camera.background_color = WHITE

        # 朝向的坐标系。
        self.axis_z_orientation = np.array([np.sin(self.orientation_phi)*np.cos(self.orientation_theta),
                                     np.sin(self.orientation_phi)*np.sin(self.orientation_theta),
                                     np.cos(self.orientation_phi)])
        self.axis_y_orientation = Z_AXIS - self.axis_z_orientation*(self.axis_z_orientation.dot(Z_AXIS))
        self.axis_x_orientation = np.cross(self.axis_y_orientation,self.axis_z_orientation)
        # 获取旋转轴及矩阵。
        self.orientation_rotate_matrix,\
            self.orientation_rotate_alpha,\
            self.orientation_rotate_beta,\
            self.orientation_rotate_gamma \
            = getRotateMatrix(self.axis_z_orientation,
                              self.axis_x_orientation,
                              self.axis_y_orientation)
        self.orientation_rotate_axis,self.orientation_rotate_angle = getRotateAxis(self.orientation_rotate_matrix)

        # 四棱锥坐标系，它最初与静止坐标系重合
        axis_scale = 2
        self.axisZ_pyramid = Arrow3D(start=ORIGIN,end=Z_AXIS*axis_scale,color=BLUE_A,resolution=8) 
        self.axisY_pyramid = Arrow3D(start=ORIGIN,end=Y_AXIS*axis_scale,color=GREEN_A,resolution=8)
        self.axisX_pyramid = Arrow3D(start=ORIGIN,end=X_AXIS*axis_scale,color=RED_A,resolution=8)
        # 不指定resolution会非常慢

        # 静止坐标系
        self.axisZ = Arrow3D(start=ORIGIN,end=Z_AXIS*axis_scale,color=BLUE,resolution=8) 
        self.axisY = Arrow3D(start=ORIGIN,end=Y_AXIS*axis_scale,color=GREEN,resolution=8)
        self.axisX = Arrow3D(start=ORIGIN,end=X_AXIS*axis_scale,color=RED,resolution=8)

        self.axisZ_origin = self.axisZ_pyramid.copy()
        self.axisX_origin = self.axisX_pyramid.copy()
        self.axisY_origin = self.axisY_pyramid.copy()


    def set_rotation(self):
        # 转动轴
        alpha,beta,gamma = np.array([80,40,-50])*DEGREES
        rotate_beta_matrix = rotation_matrix(beta,Y_AXIS)
        z_p = rotate_beta_matrix.dot(Z_AXIS)
        rotate_alpha_matrix = rotation_matrix(alpha,Z_AXIS)
        z_pp = rotate_alpha_matrix.dot(z_p)
        rotate_gamma_matrix = rotation_matrix(gamma,z_pp)
        rotate_matrix = rotate_gamma_matrix.dot(rotate_alpha_matrix.dot(rotate_beta_matrix))
        vector,angle = getRotateAxis(rotate_matrix)

        # self.axis_n = np.array([4,-0.5,2.5])
        # self.rotate_angle = -40*DEGREES
        self.axis_n = np.real(vector)
        self.rotate_angle = np.real(angle)

        self.axis_n = normalize(self.axis_n)
        # print("rotation axis is",self.axis_n,"rotation angle in degrees is",self.rotate_angle*DEGREES)
        self.vector_n_scale = 1.5
        self.vector_n = Arrow3D(start=ORIGIN,end=self.axis_n*self.vector_n_scale,resolution=8,color=GRAY)
        self.rotate_matrix = rotation_matrix(self.rotate_angle,self.axis_n)

        self.x_axis_p_end = self.rotate_matrix.dot(self.axisX_pyramid.get_end())
        self.y_axis_p_end = self.rotate_matrix.dot(self.axisY_pyramid.get_end())
        self.z_axis_p_end = self.rotate_matrix.dot(self.axisZ_pyramid.get_end())

        matrix,self.rotate_alpha,self.rotate_beta,self.rotate_gamma = getRotateMatrix(self.z_axis_p_end,self.x_axis_p_end,self.y_axis_p_end)

        self.rotate_beta_matrix = rotation_matrix(self.rotate_beta,Y_AXIS)
        self.rotate_alpha_matrix = rotation_matrix(self.rotate_alpha,Z_AXIS)
        axis_z_p = self.rotate_matrix.dot(Z_AXIS)
        self.rotate_gamma_matrix = rotation_matrix(self.rotate_gamma,axis_z_p)

    def build_labels(self):
        # 转动轴的标签
        self.label_n = MathTex("n")
        self.label_n.move_to(self.axis_n*(self.vector_n_scale+0.3))
        self.label_n.color = BLACK
        self.label_n.rotate(self.orientation_rotate_angle,self.orientation_rotate_axis)

    def set_axis_after_rotation(self):
        # 转动后的坐标系的端点，用于画圈圈
        # 静止坐标系绕y轴旋转beta角
        self.axisZ3 = self.axisZ.copy()
        self.axisZ3.color = GRAY
        self.axisZ3_end = self.rotate_beta_matrix.dot(self.axisZ.get_end())
        self.axisX3_end = self.rotate_beta_matrix.dot(self.axisX.get_end())
        self.axisY3_end = self.rotate_beta_matrix.dot(self.axisY.get_end())

        self.axisX4_end = self.rotate_alpha_matrix.dot(self.axisX3_end)
        self.axisY4_end = self.rotate_alpha_matrix.dot(self.axisY3_end)


    def build_axis_loops(self):
        # xy平面圈圈
        max_radius = 2.2
        radius_range = [max_radius-0.02,max_radius]
        self.xy_loop = Sector(radius_range=radius_range,
            resolution=8,fill_opacity=1,checkerboard_colors=None, fill_color=ORANGE,stroke_width=0)

        self.xy_p_loop = self.xy_loop.copy().rotate(self.rotate_angle,self.axis_n)

        self.xz_loop = Sector(radius_range=radius_range,
            normal_axis=Y_AXIS,start_direction=Z_AXIS,
            resolution=8,fill_opacity=1,checkerboard_colors=None, fill_color=ORANGE,stroke_width=0)

def getRotateMatrix(axis_z=np.array([0,0,1]),axis_x=np.array([1,0,0]),axis_y=np.array([0,1,0])):
    # 由通常的坐标系转到 x'y'z'，求旋转矩阵
    axis_x = axis_x/LA.norm(axis_x)
    axis_y = axis_y/LA.norm(axis_y)
    axis_z = axis_z/LA.norm(axis_z)
    # print("axes are ",axis_x,axis_y,axis_z)
    beta = np.arccos(Z_AXIS.dot(axis_z))
    # print("beta =",beta,"in degrees =",beta/DEGREES)
    R1 = rotation_matrix(beta,Y_AXIS)
    # print("R1 =\n",R1)
    Z_AXIS_1 = R1.dot(Z_AXIS)
    X_AXIS_1 = R1.dot(X_AXIS)
    # print("Z_AXIS after R1 =",Z_AXIS_1)
    # print("X_AXIS after R1 =",X_AXIS_1)
    # print("axis_z-Z_AXIS_1 =",axis_z-Z_AXIS_1)

    if beta:
        sinalpha = LA.norm(axis_z-Z_AXIS_1)/2/np.sin(beta)
        alpha = 2*np.arcsin(sinalpha)
        if axis_z.dot(np.cross(Z_AXIS,Z_AXIS_1))<0:
            alpha = -alpha

        # print("alpha =",alpha,"in degrees =",alpha/DEGREES)
        R2 = rotation_matrix(alpha,Z_AXIS)
    else:
        alpha = None
        # print("beta=0, so no need to compute alpha.")
        R2 = np.eye(len(X_AXIS))
    # if bool(axis_x):
    # print("R2 =\n",R2)
    X_AXIS_2 = R2.dot(X_AXIS_1)
    Z_AXIS_2 = R2.dot(Z_AXIS_1)
    # print("Z_AXIS after R2 =",Z_AXIS_2)
    # print("X_AXIS after R2 =",X_AXIS_2)

    gamma = np.arccos(X_AXIS_2.dot(axis_x))
    # print("corss(axis_z,axis_x) =",np.cross(axis_z,axis_x))
    if X_AXIS_2.dot(np.cross(axis_z,axis_x))>0:
        gamma = -gamma
    # elif bool(axis_x):
    #     Y_AXIS_2 = R2.dot(R1.dot(Y_AXIS))
    #     gamma = np.arccos(Y_AXIS.dot(axis_y)/LA.norm(Y_AXIS)/LA.norm(Y_AXIS_2))
    # else:
    #     raise Exception("axis x or y are needed.")
    # print("gamma =",gamma,"in degrees =",gamma/DEGREES)
    R3 = rotation_matrix(gamma,axis_z)
    # print("R3 =\n",R3)
    R = R3.dot(R2).dot(R1)
    return R,alpha,beta,gamma

def getRotateAxis(rotate_matrix:np.array,eps = 1e-5):
    # rotate_matrix: 旋转矩阵
    # 返回转轴和角度
    if (LA.det(rotate_matrix)-1)>eps:
        raise Exception("the matrix is not a rotation, whose det is 1, while that of your matrix is",LA.det(rotate_matrix))
    values,vectors = LA.eig(rotate_matrix)
    # print("all eigen values:",values)
    # print("all eigen vectors:\n",vectors)
    for i in range(len(values)):
        if abs(values[i]-1)<eps:
            # print("find eigenvalue",values[i])
            break
    else:
        raise Exception("The matrix is not a rotation.")
    trace = np.trace(rotate_matrix)
    theta = np.arccos((trace-1)/2)
    # 还需要确认旋转的方向
    vector = vectors[:,i]
    test_vector = Z_AXIS
    if LA.norm(np.cross(test_vector,vector))<eps:
        test_vector = X_AXIS
    rotate_test_vector = rotate_matrix.dot(test_vector)
    if np.cross(vector,test_vector).dot(rotate_test_vector)<0:
        theta = -theta
    return vectors[:,i],theta

def normalize(a:np.array):
    return a/LA.norm(a)

class Sector(Surface):
    def __init__(self,*args,
                    center_point:np.array=ORIGIN,    # 圆心
                    radius_range:list[float]=[0,1],  # 半径取值范围
                    angle_range:list[float]=[0,360*DEGREES],  # 角度取值范围，弧度制
                    normal_axis:np.array=Z_AXIS,     # 扇形的方向，用法向量标记
                    start_direction:np.array=X_AXIS, # 起始方向，作为角度取零时的起点，它不能与扇形的方向共线
                    eps:float=1e-5,                  # 容差
                    **kwargs
                    ):
        self.center_point = center_point
        z_axis_1 = normal_axis/LA.norm(normal_axis)
        x_axis_1 = start_direction - z_axis_1.dot(start_direction)*z_axis_1
        if LA.norm(x_axis_1)<eps:
            raise Exception("start direction and normal axis are too close.")
        x_axis_1 = x_axis_1/LA.norm(x_axis_1)
        y_axis_1 = np.cross(z_axis_1,x_axis_1)
        self.rotate_matrix,alpha,beta,gamma = getRotateMatrix(z_axis_1,x_axis_1,y_axis_1)
        super().__init__(func=self.func,u_range=radius_range,v_range=angle_range,*args,**kwargs)

    def func(self,r:float,theta:float):
        point = np.array([r*np.cos(theta),r*np.sin(theta),0])+self.center_point
        return self.rotate_matrix.dot(point)

def getRotateAxisByEuler(alpha,beta,gamma):
    rotate_beta_matrix = rotation_matrix(beta,Y_AXIS)
    z_p = rotate_beta_matrix.dot(Z_AXIS)
    rotate_alpha_matrix = rotation_matrix(alpha,Z_AXIS)
    z_pp = rotate_alpha_matrix.dot(z_p)
    rotate_gamma_matrix = rotation_matrix(gamma,z_pp)
    rotate_matrix = rotate_gamma_matrix.dot(rotate_alpha_matrix.dot(rotate_beta_matrix))
    vector,angle = getRotateAxis(rotate_matrix)
    return np.real(vector),np.real(angle)/DEGREES

def getRotateEulerByAxial(angle,axis):
    rotate_matrix = rotation_matrix(angle,axis)
    axis_x,axis_y,axis_z = map(rotate_matrix.dot,[X_AXIS,Y_AXIS,Z_AXIS])
    matrix,alpha,beta,gamma = getRotateMatrix(axis_z,axis_x,axis_y)
    return alpha,beta,gamma


# length = 100
# alphas, betas = np.meshgrid(np.linspace(0, 360, length), np.linspace(0, 360, length))
# gammas = np.linspace(0,360,30)

# for gamma in gammas:
#     gamma = gamma*DEGREES
#     x = np.zeros((length,length))
#     y = np.zeros((length,length))
#     z = np.zeros((length,length))
#     angle = np.zeros((length,length))
#     for i in range(length):
#         for j in range(length):
#             alpha = alphas[0,i]*DEGREES
#             beta = betas[0,j]*DEGREES
#             # print("alpha = %.3f, beta = %.3f, gamma = %.3f" % (alphas[0,i],betas[0,j],gamma))
#             vector,angle[i,j] = getRotateAxisByEuler(alpha,beta,gamma)
#             x[i,j],y[i,j],z[i,j] = vector

#     fig,ax = plt.subplots(2,2)
#     im = ax[0,0].pcolormesh(alphas,betas,x,shading='auto')
#     ax[0,0].set_title("x")
#     plt.colorbar(im,ax=ax[0,0])

#     im = ax[0,1].pcolormesh(alphas,betas,y,shading='auto')
#     ax[0,1].set_title("y") 
#     plt.colorbar(im,ax=ax[0,1])

#     im = ax[1,0].pcolormesh(alphas,betas,z,shading='auto')
#     ax[1,0].set_title("z")
#     plt.colorbar(im,ax=ax[1,0])

#     im = ax[1,1].pcolormesh(alphas,betas,angle,shading='auto')
#     ax[1,1].set_title("angle")
#     plt.colorbar(im,ax=ax[1,1])

#     title = "gamma=%.3f" % (gamma/DEGREES)
#     plt.suptitle(title)
#     plt.savefig(join("gamma",title+'.png'))
#     plt.close()

#     print(title,"done.")

#     # plt.show()


