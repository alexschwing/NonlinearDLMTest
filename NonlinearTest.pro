#-------------------------------------------------
#
# Project created by QtCreator 2015-07-31T10:14:42
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = NonlinearTest
CONFIG   += console
CONFIG   -= app_bundle
CONFIG += c++11
TEMPLATE = app
LIBS += -L../ -lLSDN
LIBS += -L/ais/gobi3/pkgs/cuda-7.0/lib64 -lcurand -lcublas -lcudart -lblas -lm
LIBS += -L/pkgs/mpich-3.0.4/lib/x86_64-linux-gnu -lmpich -lmpichcxx

SOURCES += main.cpp \
    dp.cpp \
    dnn.cpp

HEADERS += \
    dp.h \
    dnn.h
