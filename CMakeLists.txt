cmake_minimum_required(VERSION 3.7)
project(CMWE)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        embedding/word2vec_kernels.cc
        embedding/word2vec_ops.cc)

add_executable(CMWE ${SOURCE_FILES})