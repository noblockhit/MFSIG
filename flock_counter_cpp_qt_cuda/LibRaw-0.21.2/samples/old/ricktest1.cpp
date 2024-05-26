/* -*- C++ -*-
 * File: simple_dcraw.cpp
 * Copyright 2008-2021 LibRaw LLC (info@libraw.org)
 * Created: Sat Mar  8, 2008
 *
 * LibRaw simple C++ API:  emulates call to "dcraw  [-D]  [-T] [-v] [-e] [-4]"

LibRaw is free software; you can redistribute it and/or modify
it under the terms of the one of two licenses as you choose:

1. GNU LESSER GENERAL PUBLIC LICENSE version 2.1
   (See file LICENSE.LGPL provided in LibRaw distribution archive for details).

2. COMMON DEVELOPMENT AND DISTRIBUTION LICENSE (CDDL) Version 1.0
   (See file LICENSE.CDDL provided in LibRaw distribution archive for details).


 */
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "libraw/libraw.h"

#include <QApplication>
#include <QMainWindow>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QLabel>
#include <QVBoxLayout>
#include <QResizeEvent>
#include <QPixmap>
#include <QFileDialog>
#include <QDebug>
#include <QLibrary>

#ifndef LIBRAW_WIN32_CALLS
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#endif

#ifdef LIBRAW_WIN32_CALLS
#define snprintf _snprintf
#endif

int my_progress_callback(void *unused_data, enum LibRaw_progress state,
                         int iter, int expected)
{
  if (iter == 0)
    printf("CB: state=%x, expected %d iterations\n", state, expected);
  return 0;
}

char *customCameras[] = {
    (char *)"43704960,4080,5356, 0, 0, 0, 0,0,148,0,0, Dalsa, FTF4052C Full,0",
    (char *)"42837504,4008,5344, 0, 0, 0, 0,0,148,0,0,Dalsa, FTF4052C 3:4",
    (char *)"32128128,4008,4008, 0, 0, 0, 0,0,148,0,0,Dalsa, FTF4052C 1:1",
    (char *)"24096096,4008,3006, 0, 0, 0, 0,0,148,0,0,Dalsa, FTF4052C 4:3",
    (char *)"18068064,4008,2254, 0, 0, 0, 0,0,148,0,0,Dalsa, FTF4052C 16:9",
    (char *)"67686894,5049,6703, 0, 0, 0, 0,0,148,0,0,Dalsa, FTF5066C Full",
    (char *)"66573312,4992,6668, 0, 0, 0, 0,0,148,0,0,Dalsa, FTF5066C 3:4",
    (char *)"49840128,4992,4992, 0, 0, 0, 0,0,148,0,0,Dalsa, FTF5066C 1:1",
    (char *)"37400064,4992,3746, 0, 0, 0, 0,0,148,0,0,Dalsa, FTF5066C 4:3",
    (char *)"28035072,4992,2808, 0, 0, 0, 0,0,148,0,0,Dalsa, FTF5066C 16:9",
    NULL};

int main(int ac, char *av[])
{
  QApplication app(ac, av);


  LibRaw rawProcessor;
  return 0;
}
