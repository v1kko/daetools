#ifndef PY_EXAMPLE_H
#define PY_EXAMPLE_H

#include "variable_types.h"

/******************************************************************
	leftModel
*******************************************************************/
class leftModel : public daeModel
{
	daeDeclareDynamicClass(leftModel)
public:
	daeVariable	 var;

public:
	leftModel(string strName, daeModel* pParent = NULL, string strDescription = "") : daeModel(strName, pParent, strDescription)
	{
		AddVariable(var, "var", no_type);
	}

	void DeclareEquations(void)
	{
		daeEquation* pFn;

		pFn = CreateEquation("Distributed equation");
		pFn->SetResidual( var() - 5.0 );
	}
};

/******************************************************************
	testModel
*******************************************************************/
class testModel : public daeModel
{
	daeDeclareDynamicClass(testModel)
public:
	daeDomain    x;
	daeVariable	 v;
	daeVariable	 b;

public:
	testModel(string strName, daeModel* pParent = NULL, string strDescription = "") : daeModel(strName, pParent, strDescription)
	{
		AddDomain(x, "x");
		
		AddVariable(b, "b", no_type);
		
		v.DistributeOnDomain(x);
		AddVariable(v, "v", no_type);
	}

	void DeclareEquations(void)
	{
		daeDEDI *nx;
		daeEquation* pFn;
		
		pFn = CreateEquation("b_equation");
		pFn->SetResidual( b() - 100 );

		pFn = CreateEquation("v_equation");
		nx = pFn->DistributeOnDomain(x, eClosedClosed);
		pFn->SetResidual( v(nx) + b() );
	}
};

/******************************************************************
	simTest
*******************************************************************/
class simPython : public daeDynamicSimulation
{
public:
	simPython(void) : m("testModel")
	{
		SetModel(&m);
	}

public:
	void SetUpParametersAndDomains(void)
	{
		m.x.CreateArray(10);
	}

	void SetUpVariables(void)
	{
	}

protected:
	testModel m;
};

#endif // PY_EXAMPLE_H
