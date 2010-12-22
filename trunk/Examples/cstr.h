#ifndef DAE_EXAMPLE_CSTR_H
#define DAE_EXAMPLE_CSTR_H

#include "variable_types.h"

/******************************************************************
	CSTR model
*******************************************************************/
class modCSTR : public daeModel
{
	daeDeclareDynamicClass(modCSTR)
public:
	daeVariable	T;
	daeVariable	Tc;
	daeVariable	a;
	daeVariable	r;
	daeVariable	k;
	daeVariable	x;
	daeVariable	Qjacket;
	daeVariable	dTadiabatic;
	daeVariable	da;

	daeParameter V;
	daeParameter ro;
	daeParameter cp;
	daeParameter U;
	daeParameter A;
	daeParameter Text;
	daeParameter dHr;

public:
	modCSTR(string strName, daeModel* pParent = NULL, string strDescription = "") : daeModel(strName, pParent, strDescription)
	{
		AddVariable(T,				"T",			no_type);
		AddVariable(Tc,				"Tc",			no_type);
		AddVariable(a,				"a",			no_type);
		AddVariable(r,				"r",			no_type);
		AddVariable(k,				"k",			no_type);
		AddVariable(x,				"x",			no_type);
		AddVariable(Qjacket,		"Qjacket",		no_type);
		AddVariable(dTadiabatic,	"dTadiabatic",	no_type);
		AddVariable(da,				"da",			no_type);

		AddParameter(V,		"V",	eReal);
		AddParameter(ro,	"ro",	eReal);
		AddParameter(cp,	"cp",	eReal);
		AddParameter(U,		"U",	eReal);
		AddParameter(A,		"A",	eReal);
		AddParameter(Text,	"Text",	eReal);
		AddParameter(dHr,	"dHr",	eReal);
	}

	void DeclareEquations(void)
	{
		daeEquation* pFn;

		pFn = CreateEquation("Heat_balance");
		pFn->SetResidual( T.dt() 
			            + dHr() * r() / (ro() * cp()) 
					    + U() * A() * (T() - Text()) / (V() * ro() * cp()) 
					    );

		pFn = CreateEquation("Qjacket");
		pFn->SetResidual( Qjacket() 
					 + U() * A() * (T() - Text()) / (V() * ro() * cp()) 
					 );

		pFn = CreateEquation("dTadiabatic");
		pFn->SetResidual( dTadiabatic() 
			         + dHr() * 1900 / (ro() * cp()) 
					 );

		pFn = CreateEquation("T_celsius");
		pFn->SetResidual( Tc() - T() + 273 );

		pFn = CreateEquation("Reaction_rate");
		pFn->SetResidual( r() + k() * a() );

		pFn = CreateEquation("Mass_balance");
		pFn->SetResidual( a.dt() + k() * a() );

		pFn = CreateEquation("Mass_balance");
		pFn->SetResidual( da() - a.dt() );

		pFn = CreateEquation("k");
		pFn->SetResidual( k() - 3.7E8 * exp(-6000 / T()) );

		pFn = CreateEquation("xa");
		pFn->SetResidual( x() - a() / 1900 );
	}
};

/******************************************************************
	Simulation CSTR
*******************************************************************/
class simCSTR : public daeSimulation
{
public:
	simCSTR(void) : cb("CSTR")
	{
		SetModel(&cb);
	}

public:
	void SetUpParametersAndDomains(void)
	{
		cb.V.SetValue(1);
		cb.ro.SetValue(820);
		cb.cp.SetValue(3400);
		cb.U.SetValue(1100);
		cb.A.SetValue(4.68);
		cb.Text.SetValue(273 + 80);
		cb.dHr.SetValue(-108000);
	}

	void SetUpVariables(void)
	{
		cb.T.SetInitialCondition(273 + 25);
		cb.a.SetInitialCondition(1900);
	}

protected:
	modCSTR cb;
};


#endif
