'''
Copyright (C) 2021-2023 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''


from typing import Optional


def flops_to_string(flops: int, units: Optional[str] = None, precision: int = 2) -> str:
    """
    Converts integer MACs representation to a readable string.

    :param flops: Input MACs.
    :param units: Units for string representation of MACs (GMac, MMac or KMac).
    :param precision: Floating point precision for representing MACs in
        given units.
    """
    if units is None:
        if flops // 10**9 > 0:
            return str(round(flops / 10.**9, precision)) + ' GMac'
        elif flops // 10**6 > 0:
            return str(round(flops / 10.**6, precision)) + ' MMac'
        elif flops // 10**3 > 0:
            return str(round(flops / 10.**3, precision)) + ' KMac'
        else:
            return str(flops) + ' Mac'
    else:
        if units == 'GMac':
            return str(round(flops / 10.**9, precision)) + ' ' + units
        elif units == 'MMac':
            return str(round(flops / 10.**6, precision)) + ' ' + units
        elif units == 'KMac':
            return str(round(flops / 10.**3, precision)) + ' ' + units
        else:
            return str(flops) + ' Mac'


def params_to_string(params_num: int, units: Optional[str] = None,
                     precision: int = 2) -> str:
    """
    Converts integer params representation to a readable string.

    :param flops: Input number of parameters.
    :param units: Units for string representation of params (M, K or B).
    :param precision: Floating point precision for representing params in
        given units.
    """
    if units is None:
        if params_num // 10 ** 6 > 0:
            return str(round(params_num / 10 ** 6, precision)) + ' M'
        elif params_num // 10 ** 3:
            return str(round(params_num / 10 ** 3, precision)) + ' k'
        else:
            return str(params_num)
    else:
        if units == 'M':
            return str(round(params_num / 10.**6, precision)) + ' ' + units
        elif units == 'K':
            return str(round(params_num / 10.**3, precision)) + ' ' + units
        elif units == 'B':
            return str(round(params_num / 10.**9, precision)) + ' ' + units
        else:
            return str(params_num)
