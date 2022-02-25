FROM ros:melodic-perception-bionic
# uJCpNeVz1Q3SaowTUyxF
SHELL ["/bin/bash", "-c"]
ENV HOSTNAME=localhost

RUN echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros-latest.list
RUN apt-get update && apt-get -qq -y --allow-unauthenticated install git vim python-catkin-tools python-pip \
	ros-melodic-mongodb-store ros-melodic-ros-numpy python-backports.functools-lru-cache tmux
WORKDIR /home/v4r/catkin_ws/src
RUN git clone https://github.com/Sasha-ObjectMatching-Pipeline/table_extractor
RUN git clone https://github.com/IFL-CAMP/tf_bag.git
RUN pip install pyrsistent==0.16.1 transforms3d==0.3.1 open3d_ros_helper==0.2.0.3
RUN pip install open3d==0.9.0.0
RUN pip install scikit-image==0.14.5
WORKDIR /home/v4r/catkin_ws/
RUN . /opt/ros/melodic/setup.sh && catkin build
RUN sysctl -w net.ipv6.conf.all.forwarding=1

