clang++ -std=c++11 orb_extractor.cc orb_params.cc orb_extractor_node.cc main.cc `pkg-config --libs --cflags opencv` -lm
