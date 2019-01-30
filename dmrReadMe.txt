Overview:

gDMR is a parallelized GPU deflated minimal residual (DMR) solver for iteratively computing solution
vectors given an invertible linear operator and right hand side. The program is an implementation of 
the algorithm fully described in

``Recycling Krylov subspaces and reducing deflation subspaces for solving sequence of linear
 systems'' RR-9206, Inria Paris. 2018.

 Hussam Al Daas, Laura Grigori, Pascal Hénon, Philippe Ricoux

All use cases should cite this source. The paper should also be considered as the primary resource 
for assistance in understanding gDMR. 

Briefly, the major result of this report is that substituting a portion of the standard Krylov 
basis in the GMRES algorithm for the smallest singular vectors of the linear operator can 
substantially improve performance for multiple right hand sides. Intuitively, the largest singular
vectors are quickly discovered in computing the Krylov basis, while the smallest vectors are 
comparatively more expensive. Hence, in situations requiring high accuracy, or many iterations,
by saving a best approximation of this subspace over multiple trials or restarts performance improves.   

Use:

gDMR is designed to be combined with external code describing the linear system (linear operator),
complied with gDMR to create a shared library, and a control program, linked against this library. 
An example showing this expected use is included as the progDMR.c and testLS.cu files. To build the 
complete system in a clean directory type 

	make gDMR 

generating the shared library libgDMR.so, combining gDMR.cu with testLS.cu, then 

	make progDMR

creating the control program and linking it against the shared library containing the GPU code. 

Conventions:

All global variables are terminated with DMR to help avoid namespace overlap with linked code.

Codomain vectors (right hand sides) are referred to as images and domain vectors as source. 

The itrVecDMR pointer array is the most frequently used workspace, but genWorkVecDMR and 
sumWorkVecDMR are also occasionally used. In making any alterations to gDMR we suggest sticking to 
this convention to avoid asynchronous errors.

Notes:

Ideally, an orthogonal target basis of deflation vectors can be generated from the singular 
value decomposition of the Hessenberg without any calls to the linear operator. However, in
practice, this results in a build up of numerical error which eventually destroys fidelity. For this
reason, gDMR generates target side basis vectors using the same Arnoldi scheme as the Krylov space.
It is likely that a small, but noticeable, performance improvement could be achieved by switching 
between these two strategies after some operator dependent number of iterations.  

Basis vectors in gDMR are stored to split work between multiple devices as quickly as possible. When
iteratively constructing the Krylov basis, this means that adjacent vectors in memory do not 
correspond to subsequent iterations, and in turn, without alteration, the Hessenberg would have a 
non-standard form. To avoid this situation, primarily with an eye towards debugging, projection
coefficients in gDMR are always reorder using the reorderHessCoeffs device kernel. This reordering
is unessential to the  overall algorithm, after appropriate modifications, and could be removed. 
However, as we do not expect that this produces a tangible difference in performance reordering is 
presently included.

gDMR currently accepts single blocks and threads per block GPU settings, which are used whenever 
a gDMR kernel is called. It is likely that better efficiency can be achieved by including further 
kernel control settings based on the number of elements being operated on, type of device, and 
specific function called.
