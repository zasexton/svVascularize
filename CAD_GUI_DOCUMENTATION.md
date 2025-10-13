# svVascularize CAD-Style GUI Documentation

## Overview

The svVascularize CAD GUI provides a professional, engineering-focused interface similar to FreeCAD, SolidWorks, and AutoCAD. This interface is designed specifically for fabrication professionals who need a familiar CAD-like environment for vascular model manipulation.

## Key Features

### Professional CAD Interface
- **Dark Theme**: Professional dark gray background (#353535) standard in CAD applications
- **Dockable Panels**: All side panels can be moved, resized, or hidden
- **Multiple Toolbars**: Organized by function (File, View, Generation)
- **Model Tree**: Hierarchical object browser showing all scene elements
- **Multi-section Status Bar**: Real-time info on view, objects, and vessel counts

### Familiar CAD Conventions
- **Standard Keyboard Shortcuts**: V,F for fit view, V,I for isometric, etc.
- **Tool-based Workflow**: Click tool → Click in viewport (standard CAD pattern)
- **Object Visibility Controls**: Check/uncheck items in model tree
- **Neutral Color Scheme**: Professional grays, minimal color distraction
- **Technical Typography**: Clear, readable fonts optimized for technical work

## Interface Layout

```
┌──────────────────────────────────────────────────────────────────┐
│ Menu Bar [File | View | Generate | Help]                        │
├──────────────────────────────────────────────────────────────────┤
│ File Toolbar    [📂 💾 ⤴]                                       │
│ View Toolbar    [⛶ ▦ ⬜ ▢ ▣ ◎]                                 │
│ Generate Toolbar [● → ▶]                                        │
├─────────┬────────────────────────────────────────┬───────────────┤
│ Model   │                                        │ Properties   │
│ Tree    │                                        │              │
│         │                                        │ ┌──────────┐ │
│ □ Scene │        3D Viewport                     │ │Generation│ │
│ ├ ▦ Dom │      (VTK/PyVista)                    │ │  Params  │ │
│ ├ ● Pts │                                        │ └──────────┘ │
│ ├ 🌳 Tr │                                        │ ┌──────────┐ │
│ └ ♣ For │                                        │ │ Start    │ │
│         │                                        │ │  Points  │ │
│         │                                        │ └──────────┘ │
│         │                                        │              │
├─────────┴────────────────────────────────────────┴───────────────┤
│ Status: Ready | View: Perspective | Objects: 1 | Vessels: 0     │
└──────────────────────────────────────────────────────────────────┘
```

## Panels

### 1. Model Tree (Left Panel)
**Purpose**: Hierarchical view of all objects in the scene

**Features**:
- Checkboxes to toggle visibility
- Expand/collapse groups
- Select objects
- See object properties

**Structure**:
```
□ Scene
├ ▦ Domain (loaded)
├ ● Start Points
│  ├ ● Point 0
│  └ ● Point 1
├ 🌳 Trees
│  └ 🌳 Tree 0 (100 vessels)
└ ♣ Forests
   └ ♣ Forest 0 (2 networks)
```

**Usage**:
- Uncheck Domain to hide the mesh (see only vessels)
- Click items to select them
- Expand/collapse categories

### 2. Properties Panel (Right Panel)
**Purpose**: Tabbed interface for parameters and configuration

**Tabs**:
1. **⚙ Generation**: Tree/forest parameters
   - Mode selection
   - Vessel count
   - Physical clearance
   - Convexity tolerance
   - Random seed
   - Competition settings (forest)

2. **● Start Points**: Point management
   - Network configuration
   - Point picking
   - Direction vectors
   - Point list

**Usage**:
- Switch tabs to access different controls
- All parameters have tooltips
- Generate button at bottom of Generation tab

### 3. Information Panel (Bottom)
**Purpose**: Detailed information about selected objects

**Shows**:
- Domain statistics
- Generation progress
- Selected object properties
- Error messages

**Usage**:
- Hidden by default
- Shows automatically when domain loads
- Can be closed if not needed

## Toolbars

### File Toolbar
| Icon | Action | Shortcut | Description |
|------|--------|----------|-------------|
| 📂 | Open | Ctrl+O | Load domain mesh file |
| 💾 | Save | Ctrl+S | Save configuration |
| ⤴ | Export | - | Export generated results |

### View Toolbar
| Icon | Action | Shortcut | Description |
|------|--------|----------|-------------|
| ⛶ | Fit All | V, F | Fit all objects in view |
| ▦ | Isometric | V, I | Isometric view |
| ⬜ | Top | V, T | Top view |
| ▢ | Front | V, 1 | Front view |
| ▣ | Right | V, 3 | Right view |
| ◎ | Toggle Domain | D | Show/hide domain mesh |

### Generation Toolbar
| Icon | Action | Shortcut | Description |
|------|--------|----------|-------------|
| ● | Add Point | P | Add start point (click mode) |
| → | Add Vector | - | Add direction vector |
| ▶ | Generate | G | Generate tree/forest |

## Keyboard Shortcuts

### CAD-Standard Shortcuts
- **V, F**: Fit view (view all objects)
- **V, I**: Isometric view
- **V, T**: Top view
- **V, 1**: Front view
- **V, 3**: Right view
- **V, 6**: Back view

### File Operations
- **Ctrl+O**: Open/Load domain
- **Ctrl+S**: Save configuration
- **Ctrl+Q**: Quit application

### Tools
- **P**: Add start point tool
- **G**: Generate tree/forest
- **D**: Toggle domain visibility

### View Control (Mouse)
- **Left Drag**: Rotate view
- **Right Drag**: Pan view
- **Scroll**: Zoom in/out
- **Middle Click**: Reset view

## Status Bar Sections

The status bar has 4 sections providing real-time information:

1. **Status Message** (left): Current operation status
   - "Ready" - Idle
   - "✔ Domain loaded successfully" - Success
   - "⚠ Failed to load domain" - Error
   - "▶ Generating tree..." - Working

2. **Viewport Info**: Current view type
   - "View: Perspective"
   - "View: Isometric"
   - "View: Top"

3. **Object Count**: Number of objects in scene
   - "Objects: 0" - Empty scene
   - "Objects: 3" - Domain + 2 trees

4. **Vessel Count**: Total number of generated vessels
   - "Vessels: 0" - No generation yet
   - "Vessels: 100" - 100 vessels generated

## Workflow Examples

### Example 1: Load and View Domain
```
1. Click 📂 (Open) or press Ctrl+O
2. Select your .dmn or .vtu file
3. Status shows "✔ Domain loaded successfully"
4. Domain appears in 3D viewport
5. Model tree shows "▦ Domain (loaded)"
6. Info panel shows domain statistics
7. Press V, F to fit view if needed
```

### Example 2: Generate Single Tree
```
1. Load domain (see Example 1)
2. Click Properties panel → ⚙ Generation tab
3. Select mode: "🌳 Single Tree"
4. Set Number of Vessels: 100
5. Click ● (Add Point) toolbar button or press P
6. Click on domain mesh to place start point
7. Point appears in Model Tree under "● Start Points"
8. Click ▶ (Generate) or press G
9. Progress bar shows generation
10. Tree appears in viewport
11. Model Tree shows "🌳 Tree 0 (100 vessels)"
12. Status bar shows "Vessels: 100"
```

### Example 3: Change View
```
1. Press V, then I for isometric view
2. OR click ▦ icon in View toolbar
3. OR use mouse: left-drag to rotate
4. Press V, F to reset to fit all
5. Press D to toggle domain visibility
```

### Example 4: Hide Domain to See Vessels
```
Method 1 (Model Tree):
1. In Model Tree, find "▦ Domain"
2. Click the checkbox to uncheck it
3. Domain becomes invisible

Method 2 (Toolbar):
1. Click ◎ icon in View toolbar
2. OR press D

Method 3 (Menu):
1. View menu → Toggle Domain Visibility
```

### Example 5: Customize Workspace
```
1. Drag dock widget title bars to move panels
2. Double-click title bar to float panel
3. Right-click toolbar → Uncheck to hide
4. View menu → Panels → Select which to show
5. View menu → Toolbars → Select which to show
6. Layout auto-saves on exit
```

## Color Scheme

### Background Colors
- **Main Background**: #353535 (dark gray)
- **Panels**: #2A2A2A (darker gray)
- **Surfaces**: #2C2C2C (panel background)
- **Toolbar**: #3C3C3C (lighter gray)

### Text Colors
- **Primary Text**: #E0E0E0 (light gray)
- **Secondary Text**: #B0B0B0 (muted gray)
- **Accent Text**: #4CAF50 (green - for active items)

### Accent Colors
- **Primary Blue**: #5A7FA8 (muted blue - selections)
- **Engineering Orange**: #E67E22 (actions)
- **Success Green**: #4CAF50
- **Warning Orange**: #FF9800
- **Error Red**: #E74C3C

### Why Dark Theme?
- **Reduces eye strain** during long CAD sessions
- **Industry standard** for professional CAD/CAM software
- **Better contrast** with colored 3D objects
- **Focus on content** not interface
- **Familiar** to fabrication professionals

## Comparison: CAD vs Modern Style

| Feature | CAD Style | Modern Style |
|---------|-----------|--------------|
| Theme | Dark (#353535) | Light (#FAFAFA) |
| Layout | Toolbars + Docks | Header + Splitter |
| Navigation | Tools + Shortcuts | Buttons |
| Color | Neutral grays | Colorful (blue/orange) |
| Target User | Engineers/Fabricators | General users |
| Inspiration | FreeCAD, SolidWorks | Material Design |
| Object Browser | Hierarchical tree | Point list |
| Status Bar | Multi-section | Single message |
| Shortcuts | CAD-standard (V,F etc) | Standard (Ctrl+) |

## Switching Between Styles

### From Python:
```python
from svv.visualize.gui import launch_gui

# CAD style (default)
launch_gui(style='cad')

# Modern style
launch_gui(style='modern')
```

### From Command Line:
```bash
# CAD style
python -m svv.visualize.gui

# To use modern style, set environment variable:
export SVV_GUI_STYLE=modern
python -m svv.visualize.gui
```

## Tips for CAD Users

### 1. Use Keyboard Shortcuts
- Memorize view shortcuts: V,F V,I V,T V,1 V,3
- Use P for point tool, G for generate
- D is fastest way to toggle domain

### 2. Customize Your Workspace
- Rearrange panels to your preference
- Hide panels you don't use
- Float panels on second monitor

### 3. Use Model Tree
- Uncheck domain to see vessels clearly
- Check/uncheck to compare before/after
- Organize complex scenes

### 4. Multi-Monitor Setup
- Float Properties panel to second monitor
- Keep Model Tree on left of main display
- Maximize 3D viewport space

### 5. Fit View Often
- Press V,F after loading domain
- Press V,F after generating vessels
- Use to find lost objects

## Troubleshooting

### Problem: Dark theme is hard to see
**Solution**: The modern style has a light theme. Use `launch_gui(style='modern')`

### Problem: Toolbars taking too much space
**Solution**: View menu → Toolbars → Uncheck ones you don't need

### Problem: Can't find a panel
**Solution**: View menu → Panels → Check to show it again

### Problem: Lost in 3D space
**Solution**: Press V, F to fit all objects in view

### Problem: Domain blocking view of vessels
**Solution**: Press D or uncheck Domain in Model Tree

### Problem: Wrong keyboard shortcuts
**Solution**: CAD uses two-key shortcuts (V, F not VF). Press V, release, then press F.

## Advanced Features

### Docking System
- Drag panel title bars to move
- Drop near edges to dock
- Double-click title to float
- Close button hides (use View menu to show again)

### Tab Groups
- Right-click panel title → Tabify
- Stack multiple panels in same area
- Switch with tabs at bottom

### Toolbar Customization
- Drag toolbars to reorder
- Drag to side/bottom edges to dock there
- Double-click empty space to float

### Status Bar Information
- Click status sections for details
- Hover for tooltips
- Right-click to customize

## Integration with Fabrication Workflow

### 1. Import CAD Model
```
1. Prepare mesh in your CAD software (SolidWorks, etc.)
2. Export as .vtu or .dmn format
3. Open in svVascularize: Ctrl+O
4. Generate vasculature
```

### 2. Export for Manufacturing
```
1. Generate vessels
2. Click ⤴ (Export) in toolbar
3. Choose format for your printer/cutter
4. Import into CAM software
```

### 3. Parametric Workflow
```
1. Design domain in CAD
2. Export → Generate in svVascularize
3. If unsatisfied, adjust CAD parameters
4. Re-export → Re-generate
5. Iterate until optimal
```

## Best Practices for Fabrication

### 1. Start Simple
- Load domain and check it fits viewport
- Generate with low vessel count (50-100) first
- Verify start points are correct
- Scale up to full resolution

### 2. Check Clearances
- Use Physical Clearance parameter
- Ensure vessels don't intersect domain walls
- Verify manufacturability

### 3. Document Your Work
- Save configurations with descriptive names
- Note parameter values that work well
- Screenshot successful generations

### 4. Optimize for Manufacturing
- Consider your printer/cutter resolution
- Set vessel count appropriate for scale
- Test with clearance before full generation

## Technical Specifications

### System Requirements
- **OS**: Windows 10+, macOS 10.14+, Linux
- **RAM**: 4GB minimum, 8GB+ recommended
- **GPU**: Any with OpenGL 3.3+ (software fallback available)
- **Display**: 1920x1080 or higher recommended

### File Formats Supported
- **Input**: .dmn, .vtu (domain meshes)
- **Output**: .json (configuration), various mesh formats

### Performance
- **Small models** (<1000 vessels): Real-time generation
- **Medium models** (1000-10000 vessels): 1-10 seconds
- **Large models** (10000+ vessels): Minutes, use progress bar

## Keyboard Reference Card

```
╔═══════════════════════════════════════════════════╗
║       svVascularize CAD Keyboard Shortcuts       ║
╠═══════════════════════════════════════════════════╣
║ File Operations                                  ║
║ Ctrl+O ............ Open/Load Domain             ║
║ Ctrl+S ............ Save Configuration           ║
║ Ctrl+Q ............ Quit                         ║
╠═══════════════════════════════════════════════════╣
║ View (Two-Key Shortcuts)                         ║
║ V, F .............. Fit All in View              ║
║ V, I .............. Isometric View               ║
║ V, T .............. Top View                     ║
║ V, 1 .............. Front View                   ║
║ V, 3 .............. Right View                   ║
╠═══════════════════════════════════════════════════╣
║ Tools                                            ║
║ P ................. Add Start Point              ║
║ G ................. Generate Tree/Forest         ║
║ D ................. Toggle Domain Visibility     ║
╠═══════════════════════════════════════════════════╣
║ Mouse Controls                                   ║
║ Left Drag ......... Rotate View                 ║
║ Right Drag ........ Pan View                    ║
║ Scroll ............ Zoom In/Out                 ║
║ Middle Click ...... Reset View                  ║
╚═══════════════════════════════════════════════════╝
```

Print this card and keep it near your workstation for quick reference!

## Summary

The svVascularize CAD GUI provides a professional, familiar interface for fabrication users:

✓ **Dark theme** standard in CAD applications
✓ **Dockable panels** for customizable workspace
✓ **Hierarchical model tree** for scene organization
✓ **CAD-standard shortcuts** (V,F for fit, etc.)
✓ **Multiple toolbars** organized by function
✓ **Multi-section status bar** with real-time info
✓ **Familiar workflow** for engineers
✓ **Professional appearance** for production use
✓ **Optimized for long work sessions**
✓ **Compatible with existing CAD/CAM workflows**

Switch between CAD and Modern styles as needed to match your workflow and preferences!
