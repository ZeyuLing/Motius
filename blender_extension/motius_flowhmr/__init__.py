# SPDX-License-Identifier: GPL-3.0-or-later
"""Blender front end; all machine-learning inference runs in external Python."""
bl_info = {
    'name': 'Motius FlowHMR', 'author': 'Motius contributors',
    'version': (0, 2, 13), 'blender': (4, 2, 0),
    'location': 'View3D > Sidebar > FlowHMR', 'category': 'Animation',
}


def register():
    from . import ui
    ui.register()


def unregister():
    from . import ui
    ui.unregister()
