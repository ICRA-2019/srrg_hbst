cmake_minimum_required(VERSION 2.8)
project(srrg_hbst)

#ds INSTALL RULE
install(DIRECTORY "srrg_hbst" DESTINATION include)

#ds UNINSTALL RULE - FOR NOW ONLY SUPPORTED ON LINUX AND NOT THE CLEAN WAY - use at own risk!
add_custom_target(uninstall COMMAND
echo "removing HBST header directory: /usr/local/include/srrg_hbst" &&
rm -r -v /usr/local/include/srrg_hbst &&
echo "removing HBST pkgconfig file: /usr/local/lib/pkgconfig/srrg_hbst.pc" &&
rm -v /usr/local/lib/pkgconfig/srrg_hbst.pc &&
echo "removing HBST CMake directory: /usr/local/share/srrg_hbst" &&
rm -r -v /usr/local/share/srrg_hbst)

#ds CATKIN: define header-only catkin package
find_package(catkin)
catkin_package(INCLUDE_DIRS .)
