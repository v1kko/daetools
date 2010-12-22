#ifndef DAE_EXAMPLE_CONDUCTION_H
#define DAE_EXAMPLE_CONDUCTION_H

#include "variable_types.h"

/******************************************************************
	CopperBlock model
*******************************************************************/
class modCopperBlock : public daeModel
{
	daeDeclareDynamicClass(modCopperBlock)
public:
// Domains
	daeDomain	x;
	daeDomain	y;
	daeDomain	z;

// State variables
	daeVariable	T;
	daeVariable	Tin;
	daeVariable	Tout;
	daeVariable	Qin;
	daeVariable	Qout;
	daeVariable	ro;
	daeVariable	cp;
	daeVariable	k;

public:
	modCopperBlock(string strName, daeModel* pParent = NULL, string strDescription = "") : daeModel(strName, pParent, strDescription)
	{
		AddDomain(x, "x");
		AddDomain(y, "y");
		AddDomain(z, "z");

		AddVariable(ro, "ro",   density);
		AddVariable(cp, "cp",   heat_capacity);
		AddVariable(k,  "k",	conductivity);

		T.DistributeOnDomain(x);
		T.DistributeOnDomain(y);
		T.DistributeOnDomain(z);
		AddVariable(T, "T", temperature);

		Tin.DistributeOnDomain(x);
		Tin.DistributeOnDomain(y);
		AddVariable(Tin, "Tin", temperature);

		Tout.DistributeOnDomain(x);
		Tout.DistributeOnDomain(y);
		AddVariable(Tout, "Tout", temperature);

		Qin.DistributeOnDomain(x);
		Qin.DistributeOnDomain(y);
		AddVariable(Qin, "Qin", heat_flux);

		Qout.DistributeOnDomain(x);
		Qout.DistributeOnDomain(y);
		AddVariable(Qout, "Qout", heat_flux);
	}

	void DeclareEquations(void)
	{
		daeDEDI *nx, *ny, *nz;
		daeEquation* pFn;

		pFn = CreateEquation("Heat_balance");
		nx = pFn->DistributeOnDomain(x, eClosedClosed);
		ny = pFn->DistributeOnDomain(y, eClosedClosed);
		nz = pFn->DistributeOnDomain(z, eOpenOpen);
		pFn->SetResidual( ro() * cp() * T.dt(nx,ny,nz) - k() * (T.d2(x, nx,ny,nz) +
			                                                    T.d2(y, nx,ny,nz) + 
															    T.d2(z, nx,ny,nz)) );

		pFn = CreateEquation("Qin");
		nx = pFn->DistributeOnDomain(x, eClosedClosed);
		ny = pFn->DistributeOnDomain(y, eClosedClosed);
		pFn->SetResidual( -k() * T.d(z, nx,ny,0) - Qin(nx,ny) );

		pFn = CreateEquation("Qout");
		nx = pFn->DistributeOnDomain(x, eClosedClosed);
		ny = pFn->DistributeOnDomain(y, eClosedClosed);
		pFn->SetResidual( -k() * T.d(z, nx,ny,5) - Qout(nx,ny) );

		pFn = CreateEquation("Tin");
		nx = pFn->DistributeOnDomain(x, eClosedClosed);
		ny = pFn->DistributeOnDomain(y, eClosedClosed);
		pFn->SetResidual( T(nx,ny,0) - Tin(nx,ny) );

		pFn = CreateEquation("Tout");
		nx = pFn->DistributeOnDomain(x, eClosedClosed);
		ny = pFn->DistributeOnDomain(y, eClosedClosed);
		pFn->SetResidual( T(nx,ny,5) - Tout(nx,ny) );
	}
};

/******************************************************************
	Simulation CopperBlock
*******************************************************************/
class simCopperBlock : public daeSimulation
{
public:
	simCopperBlock(void) : cb("CopperBlock")
	{
		SetModel(&cb);
	}

public:
	void SetUpParametersAndDomains(void)
	{
		const size_t Nx = 20;
		const size_t Ny = 20;
		const size_t Nz = 5;
		cb.x.CreateDistributed(eCFDM, 2, Nx, 0, 1);
		cb.y.CreateDistributed(eCFDM, 2, Ny, 0, 1);
		cb.z.CreateDistributed(eCFDM, 2, Nz, 0, 1);
	}

	void SetUpVariables(void)
	{
		for(size_t i = 0; i < cb.x.GetNumberOfPoints(); i++)
			for(size_t j = 0; j < cb.y.GetNumberOfPoints(); j++)
				for(size_t k = 1; k < cb.z.GetNumberOfPoints()-1; k++)
					cb.T.SetInitialCondition(i, j, k, 300);

		cb.ro.AssignValue(1000);
		cb.cp.AssignValue(4186);
		cb.k.AssignValue(400);

		real_t flux[21][21] = { 30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,
							  30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,
							  30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,
							  30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,
							  30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,
							  30,30,30,30,30,30,30,30,30,30,100,100,100,100,100,100,100,30,30,30,30,
							  30,30,30,30,30,30,30,30,30,30,100,150,150,150,150,150,100,30,30,30,30,
							  30,30,30,30,30,30,30,30,30,30,100,150,200,200,200,150,100,30,30,30,30,
							  30,30,30,30,30,30,30,30,30,30,100,150,200,200,200,150,100,30,30,30,30,
							  30,30,30,30,30,30,30,30,30,30,100,150,200,200,200,150,100,30,30,30,30,
							  30,30,30,30,30,30,30,30,30,30,100,150,150,150,150,150,100,30,30,30,30,
							  30,30,30,30,30,30,30,30,30,30,100,100,100,100,100,100,100,30,30,30,30,
							  30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,
							  30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,
							  30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,
							  30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,
							  30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,
							  30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,
							  30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,
							  30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,
							  30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30 };

		for(size_t i = 0; i < cb.x.GetNumberOfPoints(); i++)
			for(size_t j = 0; j < cb.y.GetNumberOfPoints(); j++)
			{
				cb.Qin.AssignValue(i, j, flux[j][i] * 1e4);
				cb.Qout.AssignValue(i, j, 50e4);
			}
	}

protected:
	modCopperBlock cb;
};


#endif
