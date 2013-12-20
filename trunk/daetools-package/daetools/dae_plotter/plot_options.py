"""
..
 **********************************************************************************
                              daePlotOptions.py
                  DAE Tools: pyDAE module, www.daetools.com
                  Copyright (C) Dragan Nikolic, 2010
 ***********************************************************************************
 DAE Tools is free software; you can redistribute it and/or modify it under the 
 terms of the GNU General Public License version 3 as published by the Free Software 
 Foundation. DAE Tools is distributed in the hope that it will be useful, but WITHOUT 
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
 PARTICULAR PURPOSE. See the GNU General Public License for more details.
 You should have received a copy of the GNU General Public License along with the
 DAE Tools software; if not, see <http://www.gnu.org/licenses/>.
 ***********************************************************************************
"""
import sys
python_major = sys.version_info[0]

try:
    from .formlayout import fedit
except ImportError as e:
    print('[daePlotOptions]: Cannot load formlayout.fedit module', str(e))

LINESTYLES = {
              '-': 'Solid',
              '--': 'Dashed',
              '-.': 'DashDot',
              ':': 'Dotted',
#              'steps': 'Steps', # BUG!!
              'none': 'None',
              }

MARKERS = {
           'none': 'None',
           'o': 'circles',
           '^': 'triangle_up',
           'v': 'triangle_down',
           '<': 'triangle_left',
           '>': 'triangle_right',
           's': 'square',
           '+': 'plus',
           'x': 'cross',
           '*': 'star',
           'D': 'diamond',
           'd': 'thin_diamond',
           '1': 'tripod_down',
           '2': 'tripod_up',
           '3': 'tripod_left',
           '4': 'tripod_right',
           'h': 'hexagon',
           'H': 'rotated_hexagon',
           'p': 'pentagon',
           '|': 'vertical_line',
           '_': 'horizontal_line',
           '.': 'dots',
           }

COLORS = {'b': '#0000ff', 'g': '#00ff00', 'r': '#ff0000', 'c': '#ff00ff',
          'm': '#ff00ff', 'y': '#ffff00', 'k': '#000000', 'w': '#ffffff'}

def col2hex(color):
    """Convert matplotlib color to hex"""
    return COLORS.get(color, color)

def figure_edit(canvas, parent=None):
    """Edit matplotlib figure options"""
    axes = canvas.axes
    sep = (None, None) # separator

    has_curve = len(axes.get_lines())>0

    # Get / General
    xmin, xmax = axes.get_xlim()
    ymin, ymax = axes.get_ylim()
    general = [('Title', axes.get_title()),
               ('Gridlines', True),
               ('Legend', True),
               #sep,
               (None, "<b>X-Axis</b>"),
               ('Min', xmin), ('Max', xmax),
               ('Label', axes.get_xlabel()),
               ('Scale', [axes.get_xscale(), 'linear', 'log']),
               #sep,
               (None, "<b>Y-Axis</b>"),
               ('Min', ymin), ('Max', ymax),
               ('Label', axes.get_ylabel()),
               ('Scale', [axes.get_yscale(), 'linear', 'log'])
               ]

    if has_curve:
        curves = []
        linestyles = list(LINESTYLES.items())
        markers = list(MARKERS.items())
        lines = axes.get_lines()
        
        for line in lines:
            label = line.get_label()
            curvedata = [
                         ('Label', label),
                         #sep,
                         (None, '<b>Line</b>'),
                         ('Style', [line.get_linestyle()] + linestyles),
                         ('Width', line.get_linewidth()),
                         ('Color', col2hex(line.get_color())),
                         #sep,
                         (None, '<b>Marker</b>'),
                         ('Style', [line.get_marker()] + markers),
                         ('Size', line.get_markersize()),
                         ('Facecolor', col2hex(line.get_markerfacecolor())),
                         ('Edgecolor', col2hex(line.get_markeredgecolor())),
                         ]
            curves.append([curvedata, label, ""])

    datalist = [(general, "Axes", "")]
    if has_curve:
        datalist.append((curves, "Curves", ""))
    result = fedit(datalist, title="Figure options", parent=parent)
    if result is None:
        return

    if has_curve:
        general, curves = result
    else:
        general, = result

    # Set / General
    title, gridlines, legend, xmin, xmax, xlabel, xscale, ymin, ymax, ylabel, yscale = general
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_title(title)
    axes.set_xlim(xmin, xmax)
    axes.set_xlabel(xlabel)
    axes.set_ylim(ymin, ymax)
    axes.set_ylabel(ylabel)

    if has_curve:
        # Set / Curves
        for index, curve in enumerate(curves):
            line = lines[index]
            label, linestyle, linewidth, color, \
                marker, markersize, markerfacecolor, markeredgecolor = curve
            line.set_label(label)
            line.set_linestyle(linestyle)
            line.set_linewidth(linewidth)
            line.set_color(color)
            if marker is not 'none':
                line.set_marker(marker)
                line.set_markersize(markersize)
                line.set_markerfacecolor(markerfacecolor)
                line.set_markeredgecolor(markeredgecolor)

    # Redraw
    canvas.draw()

def surface_edit(canvas, parent=None):
    """Edit matplotlib figure options"""
    axes = canvas.axes
    sep = (None, None) # separator

    # Get / General
    xmin, xmax = axes.get_xlim3d()
    ymin, ymax = axes.get_ylim3d()
    zmin, zmax = axes.get_zlim3d()
    general = [#('Title', axes.get_title()),
               #sep,
               ('Color map', ['jet', 'autumn', 'bone', 'cool', 'copper', 'flag', 'gray', 'hot', 'hsv', 'jet', 'pink', 'prism', 'spring', 'summer', 'winter', 'spectral']),
               #sep,
               (None, "<b>X-Axis</b>"),
               ('Min', xmin), ('Max', xmax),
               #('Label', axes.get_xlabel()),
               #sep,
               (None, "<b>Y-Axis</b>"),
               ('Min', ymin), ('Max', ymax),
               #('Label', axes.get_ylabel()),
               #sep,
               (None, "<b>Z-Axis</b>"),
               ('Min', zmin), ('Max', zmax)
               #('Label', axes.get_zlabel()),
               ]

    datalist = [(general, "Axes", "")]
    result = fedit(datalist, title="Figure options", parent=parent)
    if result is None:
        return
    
    general, = result
    
    # Set / General
    #title, xmin, xmax, xlabel, ymin, ymax, ylabel, zmin, zmax, zlabel = general
    cmap, xmin, xmax, ymin, ymax, zmin, zmax = general
    
    #axes.set_title(title)
    
    axes.set_xlim3d(xmin, xmax)
    #axes.set_xlabel(xlabel)
    
    axes.set_ylim3d(ymin, ymax)
    #axes.set_ylabel(ylabel)
    
    axes.set_zlim3d(zmin, zmax)
    #axes.set_zlabel(zlabel)

    # Redraw
    canvas.draw()

