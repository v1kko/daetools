/***********************************************************************************
*                 DAE Tools Project: www.daetools.com
*                 Copyright (C) Dragan Nikolic, 2016
************************************************************************************
DAE Tools is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License version 3 as published by the Free Software
Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with the
DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
***********************************************************************************/
#include "auxiliary.h"

void _set_matrix_item_(real_t** matrix, size_t row, size_t col, real_t value)
{
     matrix[col][row] = value;
}

bool _compare_strings_(const char* s1, const char* s2)
{
    return (strcmp(s1, s2) == 0 ? true : false);
}

void _log_message_(const char* msg)
{
    printf(msg);
}

