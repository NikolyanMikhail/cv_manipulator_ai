import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    home = os.path.expanduser('~')
    # ПУТИ
    pkg_path = os.path.join(home, 'robot_ws/src/cv_manipulator_task')
    urdf_path = os.path.join(pkg_path, 'urdf/real_robot.urdf')
    world_path = os.path.join(pkg_path, 'urdf/task_world.sdf')
    meshes_path = os.path.join(home, 'robot_ws/src/open_manipulator')

    with open(urdf_path, 'r') as f:
        robot_desc = f.read()

    # Переменная для Gazebo
    set_env = SetEnvironmentVariable(name='GZ_SIM_RESOURCE_PATH', value=meshes_path)

    # Запуск Gazebo с нашим миром
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': f'-r {world_path}'}.items()
    )

    # Спавн робота
    spawn_robot = TimerAction(
        period=2.0,
        actions=[Node(package='ros_gz_sim', executable='create', arguments=['-string', robot_desc, '-name', 'robot', '-z', '0.01'])]
    )

    # Мост
    bridge = TimerAction(
        period=4.0,
        actions=[Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                # Команды моторам (ROS 2 -> Gazebo)
                '/joint1_cmd@std_msgs/msg/Float64]gz.msgs.Double',
                '/joint2_cmd@std_msgs/msg/Float64]gz.msgs.Double',
                '/joint3_cmd@std_msgs/msg/Float64]gz.msgs.Double',
                '/joint4_cmd@std_msgs/msg/Float64]gz.msgs.Double',
                '/gripper_left_joint_cmd@std_msgs/msg/Float64]gz.msgs.Double',
                '/gripper_right_joint_cmd@std_msgs/msg/Float64]gz.msgs.Double',
                
                # ВИДЕОПОТОК С КАМЕРЫ (Gazebo -> ROS 2)
                '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image'
            ]
        )]
    )

    return LaunchDescription([set_env, gz_sim, spawn_robot, bridge])