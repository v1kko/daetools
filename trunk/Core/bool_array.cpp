#include "stdafx.h"
#include "coreimpl.h"

namespace dae 
{
namespace core 
{

daeBoolArray::daeBoolArray()
{
}

daeBoolArray::~daeBoolArray()
{
}

void daeBoolArray::OR(const daeBoolArray& rArray)
{
	daeBoolArray::size_type i;
	for(i = 0; i < size(); i++)
	{
		if(rArray[i] || at(i))
			at(i) = true;
		else
			at(i) = false;
	}
}

bool daeBoolArray::CheckOverlapping(const daeBoolArray& rArray)
{
	daeBoolArray::size_type i;
	for(i = 0; i < size(); i++)
	{
		if(rArray[i] && at(i))
			return true;
	}
	return false;
}

}
}
