#ifndef SIMPLIFY_NODE_H
#define SIMPLIFY_NODE_H

#include <boost/format.hpp>
#include "../Core/nodes.h"

namespace daetools
{
namespace core
{
adNodePtr simplify(adNodePtr node);

adNodePtr simplify(adNodePtr node)
{
    if(dynamic_cast<adUnaryNode*>(node.get()))
    {
        adUnaryNode* un = dynamic_cast<adUnaryNode*>(node.get());
        adNodePtr n_s  = simplify(un->node);

        adNode* n = n_s.get();

        // Transform an unary node (i.e. exp(constant node)) into the constant node with value: exp(node.value)
        if(dynamic_cast<adConstantNode*>(n))
        {
            adConstantNode* val = dynamic_cast<adConstantNode*>(n);
            quantity& q = val->m_quantity;

            switch(un->eFunction)
            {
                case daetools::core::eSign:
                    return adNodePtr(new adConstantNode(-q));
                case daetools::core::eSin:
                    return adNodePtr(new adConstantNode(sin(q)));
                case daetools::core::eCos:
                    return adNodePtr(new adConstantNode(cos(q)));
                case daetools::core::eTan:
                    return adNodePtr(new adConstantNode(tan(q)));
                case daetools::core::eArcSin:
                    return adNodePtr(new adConstantNode(asin(q)));
                case daetools::core::eArcCos:
                    return adNodePtr(new adConstantNode(acos(q)));
                case daetools::core::eArcTan:
                    return adNodePtr(new adConstantNode(atan(q)));
                case daetools::core::eSqrt:
                    return adNodePtr(new adConstantNode(sqrt(q)));
                case daetools::core::eExp:
                    return adNodePtr(new adConstantNode(exp(q)));
                case daetools::core::eLn:
                    return adNodePtr(new adConstantNode(log(q)));
                case daetools::core::eLog:
                    return adNodePtr(new adConstantNode(log10(q)));
                case daetools::core::eAbs:
                    return adNodePtr(new adConstantNode(abs(q)));
                case daetools::core::eCeil:
                    return adNodePtr(new adConstantNode(ceil(q)));
                case daetools::core::eFloor:
                    return adNodePtr(new adConstantNode(floor(q)));
                case daetools::core::eSinh:
                    return adNodePtr(new adConstantNode(sinh(q)));
                case daetools::core::eCosh:
                    return adNodePtr(new adConstantNode(cosh(q)));
                case daetools::core::eTanh:
                    return adNodePtr(new adConstantNode(tanh(q)));
                case daetools::core::eArcSinh:
                    return adNodePtr(new adConstantNode(asinh(q)));
                case daetools::core::eArcCosh:
                    return adNodePtr(new adConstantNode(acosh(q)));
                case daetools::core::eArcTanh:
                    return adNodePtr(new adConstantNode(atanh(q)));
                case daetools::core::eErf:
                    return adNodePtr(new adConstantNode(erf(q)));
                default:
                    ; // do nothing (previously: "return node;" thus returning unsimplified node that was simplified but discarded)
            }
        }
        return adNodePtr(new adUnaryNode(un->eFunction, n_s));
    }
    else if(dynamic_cast<adBinaryNode*>(node.get()))
    {
        adBinaryNode* bn = dynamic_cast<adBinaryNode*>(node.get());
        adNodePtr left_s  = simplify(bn->left);
        adNodePtr right_s = simplify(bn->right);

        adNode* left  = left_s.get();
        adNode* right = right_s.get();

        if(dynamic_cast<adConstantNode*>(left) && dynamic_cast<adConstantNode*>(right)) // c OP c => return a value
        {
            adConstantNode* cleft  = dynamic_cast<adConstantNode*>(left);
            adConstantNode* cright = dynamic_cast<adConstantNode*>(right);

            if(bn->eFunction == daetools::core::ePlus)
                return adNodePtr(new adConstantNode(cleft->m_quantity + cright->m_quantity));
            else if(bn->eFunction == daetools::core::eMinus)
                return adNodePtr(new adConstantNode(cleft->m_quantity - cright->m_quantity));
            else if(bn->eFunction == daetools::core::eMulti)
                return adNodePtr(new adConstantNode(cleft->m_quantity * cright->m_quantity));
            else if(bn->eFunction == daetools::core::eDivide)
                return adNodePtr(new adConstantNode(cleft->m_quantity / cright->m_quantity));

            else if(bn->eFunction == daetools::core::ePower)
                return adNodePtr(new adConstantNode(units::pow(cleft->m_quantity, cright->m_quantity)));
            else if(bn->eFunction == daetools::core::eArcTan2)
                return adNodePtr(new adConstantNode(units::atan2(cleft->m_quantity, cright->m_quantity)));
            else if(bn->eFunction == daetools::core::eMin)
                return adNodePtr(new adConstantNode(units::min(cleft->m_quantity, cright->m_quantity)));
            else if(bn->eFunction == daetools::core::eMax)
                return adNodePtr(new adConstantNode(units::max(cleft->m_quantity, cright->m_quantity)));
        }
        else if(dynamic_cast<adConstantNode*>(left) && dynamic_cast<adFloatCoefficientVariableSumNode*>(right)) // c1 OP base+sum(c*Var) => combine them
        {
            adConstantNode*                    cleft  = dynamic_cast<adConstantNode*>(left);
            adFloatCoefficientVariableSumNode* cright = dynamic_cast<adFloatCoefficientVariableSumNode*>(right);

            if(bn->eFunction == daetools::core::eMulti) // c1 * (base+sum(c*Var)) => c1*base + sum(c1*c*Var)
            {
                if(cleft->m_quantity.getValue() == 0.0)
                    return adNodePtr(new adConstantNode(0.0));

                std::map<size_t, daeFloatCoefficientVariableProduct>::iterator it;

                adFloatCoefficientVariableSumNode* fcvs = new adFloatCoefficientVariableSumNode();
                fcvs->m_sum  = cright->m_sum;
                fcvs->m_base = cright->m_base * cleft->m_quantity.getValue();

                for(it = fcvs->m_sum.begin(); it != fcvs->m_sum.end(); it++)
                {
                    daeFloatCoefficientVariableProduct& fcvp = it->second;
                    fcvp.coefficient *= cleft->m_quantity.getValue();
                }
                return adNodePtr(fcvs);
            }
            else if(bn->eFunction == daetools::core::ePlus) // c1 + (base+sum(c*Var)) => c1+base + sum(c*Var)
            {
                adFloatCoefficientVariableSumNode* fcvs = new adFloatCoefficientVariableSumNode();
                fcvs->m_sum  = cright->m_sum;
                fcvs->m_base = cright->m_base + cleft->m_quantity.getValue();

                return adNodePtr(fcvs);
            }
        }
        else if(dynamic_cast<adFloatCoefficientVariableSumNode*>(left) && dynamic_cast<adConstantNode*>(right)) // sum(c*Var) OP c1 => combine them
        {
            adFloatCoefficientVariableSumNode* cleft  = dynamic_cast<adFloatCoefficientVariableSumNode*>(left);
            adConstantNode*                    cright = dynamic_cast<adConstantNode*>(right);

            if(bn->eFunction == daetools::core::eMulti) // (base+sum(c*Var)) * c1 => c1*base + sum(c1*c*Var)
            {
                if(cright->m_quantity.getValue() == 0.0)
                    return adNodePtr(new adConstantNode(0.0));

                std::map<size_t, daeFloatCoefficientVariableProduct>::iterator it;

                adFloatCoefficientVariableSumNode* fcvs = new adFloatCoefficientVariableSumNode();
                fcvs->m_sum  = cleft->m_sum;
                fcvs->m_base = cleft->m_base * cright->m_quantity.getValue();

                for(it = fcvs->m_sum.begin(); it != fcvs->m_sum.end(); it++)
                {
                    daeFloatCoefficientVariableProduct& fcvp = it->second;
                    fcvp.coefficient *= cright->m_quantity.getValue();
                }
                return adNodePtr(fcvs);
            }
            else if(bn->eFunction == daetools::core::eDivide) // (base+sum(c*Var)) / c1 => base/c1 + sum(c1/c*Var)
            {
                // Division by zero!!!
                if(cright->m_quantity.getValue() == 0.0)
                    return adNodePtr(new adConstantNode( std::numeric_limits<double>::quiet_NaN() ));

                std::map<size_t, daeFloatCoefficientVariableProduct>::iterator it;

                adFloatCoefficientVariableSumNode* fcvs = new adFloatCoefficientVariableSumNode();
                fcvs->m_sum  = cleft->m_sum;
                fcvs->m_base = cleft->m_base / cright->m_quantity.getValue();

                for(it = fcvs->m_sum.begin(); it != fcvs->m_sum.end(); it++)
                {
                    daeFloatCoefficientVariableProduct& fcvp = it->second;
                    fcvp.coefficient /= cright->m_quantity.getValue();
                }
                return adNodePtr(fcvs);
            }
            else if(bn->eFunction == daetools::core::ePlus) // (base+sum(c*Var)) + c1 => c1+base + sum(c*Var)
            {
                adFloatCoefficientVariableSumNode* fcvs = new adFloatCoefficientVariableSumNode();
                fcvs->m_sum  = cleft->m_sum;
                fcvs->m_base = cleft->m_base + cright->m_quantity.getValue();

                return adNodePtr(fcvs);
            }
        }
        else if(dynamic_cast<adFloatCoefficientVariableSumNode*>(left) && dynamic_cast<adFloatCoefficientVariableSumNode*>(right)) // sum(c*Var) OP sum(c*Var) => combine them
        {
            adFloatCoefficientVariableSumNode* cleft  = dynamic_cast<adFloatCoefficientVariableSumNode*>(left);
            adFloatCoefficientVariableSumNode* cright = dynamic_cast<adFloatCoefficientVariableSumNode*>(right);

            if(bn->eFunction == daetools::core::ePlus) // (base1+sum(c1*Var)) + (base2+sum(c2*Var)) => base1+base2 + sum((c1+c2)*Var)
            {
                std::map<size_t, daeFloatCoefficientVariableProduct>::iterator it, it_find;

                adFloatCoefficientVariableSumNode* fcvs = new adFloatCoefficientVariableSumNode();
                fcvs->m_sum  = cleft->m_sum;
                fcvs->m_base = cleft->m_base + cright->m_base;

                for(it = cright->m_sum.begin(); it != cright->m_sum.end(); it++)
                {
                    size_t                              r_overallIndex = it->first;
                    daeFloatCoefficientVariableProduct& r_fcvp         = it->second;

                    it_find = fcvs->m_sum.find(r_overallIndex);
                    if(it_find != fcvs->m_sum.end()) // it already exists - just add the coefficient
                    {
                        daeFloatCoefficientVariableProduct& fcvp = it_find->second;
                        fcvp.coefficient += r_fcvp.coefficient;
                    }
                    else // it doesn't exist - add a new item
                    {
                        fcvs->m_sum[r_overallIndex] = r_fcvp;
                    }
                }
                return adNodePtr(fcvs);
            }
            else if(bn->eFunction == daetools::core::eMinus) // (base1+sum(c1*Var)) - (base2+sum(c2*Var)) => base1-base2 + sum((c1-c2)*Var)
            {
                std::map<size_t, daeFloatCoefficientVariableProduct>::iterator it, it_find;

                adFloatCoefficientVariableSumNode* fcvs = new adFloatCoefficientVariableSumNode();
                fcvs->m_sum  = cleft->m_sum;
                fcvs->m_base = cleft->m_base - cright->m_base;

                for(it = cright->m_sum.begin(); it != cright->m_sum.end(); it++)
                {
                    size_t                              r_overallIndex = it->first;
                    daeFloatCoefficientVariableProduct& r_fcvp         = it->second;

                    it_find = fcvs->m_sum.find(r_overallIndex);
                    if(it_find != fcvs->m_sum.end()) // it already exists - just subtract the coefficient
                    {
                        daeFloatCoefficientVariableProduct& fcvp = it_find->second;
                        fcvp.coefficient -= r_fcvp.coefficient;
                    }
                    else // it doesn't exist - add a new item
                    {
                        daeFloatCoefficientVariableProduct new_fcvp = r_fcvp;
                        new_fcvp.coefficient = -new_fcvp.coefficient;
                        fcvs->m_sum[r_overallIndex] = new_fcvp;
                    }
                }
                return adNodePtr(fcvs);
            }
        }
        else if(dynamic_cast<adConstantNode*>(left)) // if left == 0
        {
            adConstantNode* cn = dynamic_cast<adConstantNode*>(left);
            if(cn->m_quantity.getValue() == 0)
            {
                if(bn->eFunction == daetools::core::ePlus) // 0 + right => right
                    return right_s;
                else if(bn->eFunction == daetools::core::eMulti) // 0 * right => 0 (that is left)
                    return left_s;
                else if(bn->eFunction == daetools::core::eDivide) // 0 / right => 0 (that is left)
                    return left_s;
            }
        }
        else if(dynamic_cast<adConstantNode*>(right)) // if right == 0
        {
            adConstantNode* cn = dynamic_cast<adConstantNode*>(right);
            if(cn->m_quantity.getValue() == 0)
            {
                if(bn->eFunction == daetools::core::ePlus) // left + 0 => left
                    return left_s;
                else if(bn->eFunction == daetools::core::eMinus) // left - 0 => left
                    return left_s;
                else if(bn->eFunction == daetools::core::eMulti) // left * 0 => 0 (that is right)
                    return right_s;
            }
        }

        return adNodePtr(new adBinaryNode(bn->eFunction, left_s, right_s));
    }
    else
    {
        return node;
    }
}

}
}

#endif
