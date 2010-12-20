#ifndef DAE_OPTIMIZATION_H
#define DAE_OPTIMIZATION_H

#include "stdafx.h"
#include "dyn_simulation.h"
#include <stdio.h>
#include <time.h>

namespace dae
{
namespace activity
{
class DAE_ACTIVITY_API daeOptimization : public daeDynamicSimulation
{
public:
	daeOptimization(void);
	virtual ~daeOptimization(void);

public:
};

}
}
#endif
