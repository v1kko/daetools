#ifndef DAE_EXAMPLE_STRAIGHT_FIN1_H
#define DAE_EXAMPLE_STRAIGHT_FIN1_H

#include "variable_types.h"

/******************************************************************
	StraightFin1 model
*******************************************************************/
const size_t Nsf1 = 20;

class modStraightFin1 : public daeModel
{
	daeDeclareDynamicClass(modStraightFin1)
public:
	daeDomain    x;

	daeParameter d;
	daeParameter b;
	daeParameter L;
	daeParameter g;
	daeParameter h0;
	daeParameter k;
	daeParameter Tb;
	daeParameter Ts;

	daeVariable	 T;
	daeVariable	 Theta;
	daeVariable	 h;
	daeVariable	 eta;
	daeVariable	 Qbase;
	daeVariable	 Qideal;
	daeVariable	 htemp;

public:
	modStraightFin1(string strName, daeModel* pParent = NULL, string strDescription = "") : daeModel(strName, pParent, strDescription)
	{
		AddDomain(x, "x");
		
		Theta.DistributeOnDomain(x);
		AddVariable(Theta, "Theta", temperature);

		h.DistributeOnDomain(x);
		AddVariable(h, "h", heat_tr_coeff);

		htemp.DistributeOnDomain(x);
		AddVariable(htemp, "htemp", no_type);

		AddVariable(eta,    "eta",    fraction);
		AddVariable(Qbase,  "Qbase",  heat);
		AddVariable(Qideal, "Qideal", heat);

		AddParameter(d,  "d",  eReal);
		AddParameter(b,  "b",  eReal);
		AddParameter(L,  "L",  eReal);
		AddParameter(g,  "g",  eReal);
		AddParameter(h0, "h0", eReal);
		AddParameter(k,  "k",  eReal);
		AddParameter(Tb, "Tb", eReal);
		AddParameter(Ts, "Ts", eReal);
	}

	void DeclareEquations(void)
	{
		daeDEDI *nx;
		daeEquation* pFn;

		pFn = CreateEquation("Heat_balance");
		nx = pFn->DistributeOnDomain(x, eOpenOpen);
		pFn->SetResidual( Theta.d2(x, nx) - 2 * h(nx) * Theta(nx) / (k() * d()) );

		pFn = CreateEquation("BC_left");
		pFn->SetResidual( Theta(0) - (Tb() - Ts()) );

		pFn = CreateEquation("BC_right");
		pFn->SetResidual( Theta.d(x, Nsf1)  );

		pFn = CreateEquation("h(x)");
		nx = pFn->DistributeOnDomain(x, eClosedClosed);
		pFn->SetResidual( h(nx) - (g() + 1) * h0() * pow(0.5, g()) );

		pFn = CreateEquation("htemp(x)");
		nx = pFn->DistributeOnDomain(x, eClosedClosed);
		pFn->SetResidual( htemp(nx) - (*nx)() / b() );

		pFn = CreateEquation("Eta");
		pFn->SetResidual( eta() * Qideal() - Qbase() ); 

		pFn = CreateEquation("Qbase");
		pFn->SetResidual( Qbase() + k() * d() * L() * Theta.d(x, 0) );

		pFn = CreateEquation("Qbase");
		pFn->SetResidual( Qideal() - 2 * h0() * L() * b() * Theta(0) );
	}
};

/******************************************************************
	Simulation StraightFin1
*******************************************************************/
class simStraightFin1 : public daeDynamicSimulation
{
public:
	simStraightFin1(void) : sf("StraightFin1")
	{
		SetModel(&sf);
		sf.SetName("StraightFin1");
	}

public:
	void SetUpParametersAndDomains(void)
	{
		sf.x.CreateDistributed(eCFDM, 2, Nsf1, 0, 7.62E-2);

		sf.d.SetValue(0.3226E-2);
		sf.b.SetValue(7.62E-2);
		sf.L.SetValue(10E-2);
		sf.g.SetValue(4);
		sf.h0.SetValue(15);
		sf.k.SetValue(30);
		sf.Tb.SetValue(100);
		sf.Ts.SetValue(20);
	}
	void SetUpVariables(void)
	{
	}
	
protected:
	modStraightFin1 sf;
};


#endif
