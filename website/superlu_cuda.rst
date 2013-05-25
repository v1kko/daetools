*************
SuperLU CUDA
*************
..
    Copyright (C) Dragan Nikolic, 2013
    DAE Tools is free software; you can redistribute it and/or modify it under the
    terms of the GNU General Public License version 3 as published by the Free Software
    Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
    PARTICULAR PURPOSE. See the GNU General Public License for more details.
    You should have received a copy of the GNU General Public License along with the
    DAE Tools software; if not, see <http://www.gnu.org/licenses/>.

This is a new DAE Tools subproject with the aim to provide a direct
sparse linear equation solver which works with NVidia CUDA GPUs. Get the
latest source code `superlu\_cuda-0.0.1.tar.gz`_.

Current status
--------------

The starting point is **test\_superlu\_cuda.cu** file. Currently only
functions **get\_perm\_c**, **pdgstrf\_init**, **pdgstrf** and
**dgstrs** are ported (not the driver functions) and all have the prefix
gpu\_ (**gpu\_get\_perm\_c**, **gpu\_pdgstrf\_init**, **gpu\_pdgstrf**
and **gpu\_dgstrs**).

.. _superlu\_cuda-0.0.1.tar.gz: http://{{SERVERNAME}}/superlu_cuda-0.0.1.tar.gz

In superlu\_cuda everything resides on a device: matrices A, B, L, U, AC
etc and everything executes on the device. CUDA threads are started from
the host by the function **cuda\_pdgstrf**. Matrices are not copied back
and forth from/to the device (except values for **nzval**, **rowind**
and **colptr**). The memory for **nzval**, **rowind** and **colptr** is
allocated with cudaMalloc while matrices are created with
**gpu\_dCreate\_CompCol\_Matrix** and **gpu\_dCreate\_Dense\_Matrix**
kernel calls.


The main problem is with critical sections. Currently they are
implemented by using atomic operations (thanks to cvnguyen, Sarnath,
tmurray and others for excelent forum posts on `NVidia GPU Computing`_).
The synchronisation points are in: **pxgstrf\_scheduler.c** (lines 108,
275), **pdgstrf\_panel\_bmod.c** (lines 237, 282), **pmemory.c**
(function Glu\_alloc, lines 169/197 and 225/254; function DynamicSetMap
lines 302/330), **pxgstrf\_synch.c** (function NewNsuper lines 358/376).
I have problems with synchronisation within warps of a block and it
seems between blocks as well. Therefore I can run only one thread per warp (a group of 32 threads), so
I can run on my device 512/32 = 16 threads (out of 4 SMs each with 48
cores = 192). Anyway, the additional memory for iwork and dwork arrays
for a single thread is rather high (approximately 130 \* N if I am
correct) so not too many threads can be started anyway (for N=100 000
the memory per thread is >13MB).

This is the current status. The problems are the following:

-  Synchronisation between threads in a warp.
-  Synchronisation between threads in different blocks.

Any help is appreciated. You can email me at dnikolic - daetools dot
com.

.. _NVidia GPU Computing: http://forums.nvidia.com/index.php?s=35f6610fb56e3ab2e319eed132a93ef7&showforum=62



.. image:: http://sourceforge.net/apps/piwik/daetools/piwik.php?idsite=1&amp;rec=1&amp;url=wiki/
    :alt:
