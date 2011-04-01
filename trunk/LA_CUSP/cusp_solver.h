#ifndef CUSP_SOLVER_H
#define CUSP_SOLVER_H


namespace dae
{
namespace solver
{
class cusp_solver
{
public:
    cusp_solver();
	
	void ResizeMatrixAndVectors(int NNZ, int N, int* IA, int* JA);
	void SetMatrix_A(int NNZ, double* A);
	void SetVector_b(int N, double* b);
	void GetVector_x(int N, double** x);
	
	bool BuildPreconditioner();
	bool Solve();
	
protected:
	void* m_pData;
};

}
}

#endif // CUSP_SOLVER_H
