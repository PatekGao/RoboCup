CMAKE_MINIMUM_REQUIRED(VERSION 2.6.0)
PROJECT(polyclipping)

SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Release type")

SET(BUILD_SHARED_LIBS ON CACHE BOOL
    "Build shared libraries (.dll/.so) instead of static ones (.lib/.a)")
ADD_LIBRARY(polyclipping clipper.cpp)
