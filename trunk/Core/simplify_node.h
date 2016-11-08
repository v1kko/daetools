#ifndef SIMPLIFY_NODE_H
#define SIMPLIFY_NODE_H

#include <boost/format.hpp>
#include "../Core/nodes.h"

namespace dae
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
                case dae::core::eSign:
                    return adNodePtr(new adConstantNode(-q));
                case dae::core::eSin:
                    return adNodePtr(new adConstantNode(sin(q)));
                case dae::core::eCos:
                    return adNodePtr(new adConstantNode(cos(q)));
                case dae::core::eTan:
                    return adNodePtr(new adConstantNode(tan(q)));
                case dae::core::eArcSin:
                    return adNodePtr(new adConstantNode(asin(q)));
                case dae::core::eArcCos:
                    return adNodePtr(new adConstantNode(acos(q)));
                case dae::core::eArcTan:
                    return adNodePtr(new adConstantNode(atan(q)));
                case dae::core::eSqrt:
                    return adNodePtr(new adConstantNode(sqrt(q)));
                case dae::core::eExp:
                    return adNodePtr(new adConstantNode(exp(q)));
                case dae::core::eLn:
                    return adNodePtr(new adConstantNode(log(q)));
                case dae::core::eLog:
                    return adNodePtr(new adConstantNode(log10(q)));
                case dae::core::eAbs:
                    return adNodePtr(new adConstantNode(abs(q)));
                case dae::core::eCeil:
                    return adNodePtr(new adConstantNode(ceil(q)));
                case dae::core::eFloor:
                    return adNodePtr(new adConstantNode(floor(q)));
                case dae::core::eSinh:
                    return adNodePtr(new adConstantNode(sinh(q)));
                case dae::core::eCosh:
                    return adNodePtr(new adConstantNode(cosh(q)));
                case dae::core::eTanh:
                    return adNodePtr(new adConstantNode(tanh(q)));
                case dae::core::eArcSinh:
                    return adNodePtr(new adConstantNode(asinh(q)));
                case dae::core::eArcCosh:
                    return adNodePtr(new adConstantNode(acosh(q)));
                case dae::core::eArcTanh:
                    return adNodePtr(new adConstantNode(atanh(q)));
                case dae::core::eErf:
                    return adNodePtr(new adConstantNode(erf(q)));
                default:
                    return node;
            }
        }
        return node;
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

            if(bn->eFunction == dae::core::ePlus)
                return adNodePtr(new adConstantNode(cleft->m_quantity + cright->m_quantity));
            else if(bn->eFunction == dae::core::eMinus)
                return adNodePtr(new  adConstantNode(cleft->m_quantity - cright->m_quantity));
            else if(bn->eFunction == dae::core::eMulti)
                return adNodePtr(new  adConstantNode(cleft->m_quantity * cright->m_quantity));
            else if(bn->eFunction == dae::core::eDivide)
                return adNodePtr(new  adConstantNode(cleft->m_quantity / cright->m_quantity));
        }
        else if(dynamic_cast<adConstantNode*>(left)) // if left == 0
        {
            adConstantNode* cn = dynamic_cast<adConstantNode*>(left);
            if(cn->m_quantity.getValue() == 0)
            {
                if(bn->eFunction == dae::core::ePlus) // 0 + right => right
                    return right_s;
                else if(bn->eFunction == dae::core::eMulti) // 0 * right => 0 (that is left)
                    return left_s;
                else if(bn->eFunction == dae::core::eDivide) // 0 / right => 0 (that is left)
                    return left_s;
            }
        }
        else if(dynamic_cast<adConstantNode*>(right)) // if right == 0
        {
            adConstantNode* cn = dynamic_cast<adConstantNode*>(right);
            if(cn->m_quantity.getValue() == 0)
            {
                if(bn->eFunction == dae::core::ePlus) // left + 0 => left
                    return left_s;
                else if(bn->eFunction == dae::core::eMinus) // left - 0 => left
                    return left_s;
                else if(bn->eFunction == dae::core::eMulti) // left * 0 => 0 (that is right)
                    return right_s;
            }
        }

        return node;
    }
    else
    {
        return node;
    }
}

}
}

#endif
