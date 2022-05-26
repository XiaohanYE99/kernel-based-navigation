# CMake generated Testfile for 
# Source directory: /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/examples
# Build directory: /home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/build/examples
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(Blocks "/home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/build/examples/Blocks")
set_tests_properties(Blocks PROPERTIES  LABELS "medium" TIMEOUT "60")
add_test(Circle "/home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/build/examples/Circle")
set_tests_properties(Circle PROPERTIES  LABELS "medium" TIMEOUT "60")
add_test(Roadmap "/home/yxhan/yxh/kernel-based-navigation-master/RVO2-main/build/examples/Roadmap")
set_tests_properties(Roadmap PROPERTIES  LABELS "medium" TIMEOUT "60")
