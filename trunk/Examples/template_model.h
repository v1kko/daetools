#ifndef TEMPLATE_MODEL_H
#define TEMPLATE_MODEL_H

#include "variable_types.h"

/******************************************************************
	templateModel model
*******************************************************************/
class templateModel : public daeModel
{
	daeDeclareDynamicClass(templateModel)
public:
	daeDomain    domain;
	daeParameter param;
	daeVariable	 var;

public:
	void DeclareData(void)
	{
		AddDomain(domain, "domain");
		
		var.DistributeOnDomain(domain);
		AddVariable(var, "var", no_type);

		AddParameter(param, "param", eReal);

		daeModel::DeclareData();
	}

	void DeclareEquations(void)
	{
		daeDEDI *nx;
		daeEquation* pFn;

		pFn = AddEquation("Distributed equation");
		nx = pFn->DistributeOnDomain(domain, eOpenOpen);
		pFn->residual( var(nx) - param() );

		daeModel::DeclareEquations();
	}
};

/******************************************************************
	Simulation templateSimulation
*******************************************************************/
class templateSimulation : public daeDynamicSimulation
{
public:
	templateSimulation(void)
	{
		SetModel(&m);
		m.SetName("templateModel");
	}

public:
	void SetUpParametersAndDomains(void)
	{
		const size_t Nx = 10;
		m.domain.CreateDistributed(eCFDM, 2, Nx, 0, 0.1);

		m.param.SetValue(1e-3);

		daeDynamicSimulation::SetUpParametersAndDomains();
	}

	void SetUpVariables(void)
	{
		daeDynamicSimulation::SetUpVariables();
	}

protected:
	templateModel m;
};


#endif // TEMPLATE_MODEL_H
