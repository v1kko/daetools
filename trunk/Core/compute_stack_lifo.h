/***********************************************************************************
*                 DAE Tools Project: www.daetools.com
*                 Copyright (C) Dragan Nikolic
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#if !defined(C_LIFO_STACK_IMPLEMENTATION_H)
#define C_LIFO_STACK_IMPLEMENTATION_H

#include <stdlib.h>
#include <stdio.h>
#include "compute_stack.h"

typedef struct
{
    int         size;
    adouble_cs* data;
    int         top;
} lifo_stack_t;

CS_DECL int lifo_init(lifo_stack_t* stack, int stack_size)
{
    if(stack_size > 0)
    {
        stack->size = stack_size;
        stack->data = (adouble_cs*)malloc(stack_size * sizeof(adouble_cs));
    }
    else
    {
        stack->size = 0;
        stack->data = NULL;
    }
    stack->top = -1;
}

CS_DECL int lifo_free(lifo_stack_t* stack)
{
    stack->size = 0;
    free(stack->data);
    stack->top = -1;
}

CS_DECL int lifo_isempty(lifo_stack_t* stack)
{
    if(stack->top == -1)
        return 1;
    else
        return 0;
}

CS_DECL int lifo_isfull(lifo_stack_t* stack)
{
    if(stack->top == stack->size - 1)
        return 1;
    else
        return 0;
}

CS_DECL adouble_cs lifo_top(lifo_stack_t* stack)
{
#ifdef __cplusplus
    if(stack->top < 0)
        throw std::runtime_error("lifo_top: top < 0");
#endif
    return stack->data[stack->top];
}

CS_DECL void lifo_pop(lifo_stack_t* stack)
{
    if(!lifo_isempty(stack))
    {
        stack->top = stack->top - 1;
    }
    else
    {
#ifdef __cplusplus
        throw std::runtime_error("lifo_pop: stack is empty");
#endif
    }
}

CS_DECL void lifo_push(lifo_stack_t* stack, adouble_cs* item)
{
    if(!lifo_isfull(stack))
    {
        stack->top = stack->top + 1;
        adouble_copy(&stack->data[stack->top], item);
    }
    else
    {
#ifdef __cplusplus
        throw std::runtime_error("lifo_pop: stack is full");
#endif
    }
}

#endif
