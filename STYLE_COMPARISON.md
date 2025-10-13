# svVascularize GUI Style Comparison

## Quick Visual Guide

### CAD Style (Default) - For Engineers & Fabrication

```
╔═══════════════════════════════════════════════════════════════════╗
║ File | View | Generate | Help                                     ║
╠═══════════════════════════════════════════════════════════════════╣
║ [📂 💾 ⤴] [⛶ ▦ ⬜ ▢ ▣ ◎] [● → ▶]                              ║
╠══════════╦═══════════════════════════════════════════╦════════════╣
║ Model    ║                                           ║ Properties ║
║ Tree     ║                                           ║            ║
║          ║                                           ║ ┌────────┐ ║
║ □ Scene  ║          3D Viewport                      ║ │  Gen   │ ║
║ ├▦ Dom   ║         (Dark #353535)                   ║ │ Params │ ║
║ ├● Pts   ║                                           ║ └────────┘ ║
║ ├🌳 Tr   ║                                           ║ ┌────────┐ ║
║ └♣ For   ║                                           ║ │ Start  │ ║
║          ║                                           ║ │ Points │ ║
║          ║                                           ║ └────────┘ ║
╠══════════╩═══════════════════════════════════════════╩════════════╣
║ Ready | View: Perspective | Objects: 1 | Vessels: 0              ║
╚═══════════════════════════════════════════════════════════════════╝

Colors: Dark gray (#353535), neutral tones, muted accents
Layout: Dockable panels, multiple toolbars, model tree
Target: Engineers, fabrication professionals, CAD users
```

### Modern Style (Optional) - For General Users

```
╔═══════════════════════════════════════════════════════════════════╗
║ 🌳 svVascularize                                                  ║
║ Interactive vascular tree and forest generation                  ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  ┌───────────────────────────────┬───────────────────────────┐   ║
║  │                               │  ┌─────────────────────┐  │   ║
║  │                               │  │ • Start Points       │  │   ║
║  │                               │  │   & Directions       │  │   ║
║  │        3D Viewport            │  └─────────────────────┘  │   ║
║  │       (Light #FAFAFA)         │  ┌─────────────────────┐  │   ║
║  │                               │  │ ⚙ Generation        │  │   ║
║  │                               │  │   Parameters         │  │   ║
║  │                               │  │                      │  │   ║
║  │                               │  │ ▶ Generate           │  │   ║
║  └───────────────────────────────┴───────────────────────────┘   ║
╠═══════════════════════════════════════════════════════════════════╣
║ ✔ Ready                                                           ║
╚═══════════════════════════════════════════════════════════════════╝

Colors: Light background (#FAFAFA), blue/orange accents
Layout: Header, fixed splitter, side panels
Target: General users, consumer applications, demos
```

## Side-by-Side Comparison

### Visual Appearance

| Aspect | CAD Style | Modern Style |
|--------|-----------|--------------|
| **Background** | Dark gray (#353535) | Light gray (#FAFAFA) |
| **Panels** | Dark (#2A2A2A) | White (#FFFFFF) |
| **Text** | Light (#E0E0E0) | Dark (#212121) |
| **Accents** | Muted blue (#5A7FA8) | Bright blue (#2196F3) |
| **Actions** | Orange (#E67E22) | Orange (#FF5722) |
| **Overall Feel** | Technical, professional | Clean, modern |

### Interface Elements

| Element | CAD Style | Modern Style |
|---------|-----------|--------------|
| **Header** | Menu bar + toolbars | Branded header card |
| **Toolbars** | 3 functional toolbars (15 tools) | None (buttons in panels) |
| **Object Browser** | Hierarchical tree with checkboxes | Point list only |
| **Panels** | Dockable, movable, floating | Fixed position splitter |
| **Status Bar** | 4 sections with detailed info | Single message line |
| **Icons** | Technical symbols (●▦→) | Unicode emojis (🌳💾📂) |

### Keyboard Shortcuts

| Action | CAD Style | Modern Style |
|--------|-----------|--------------|
| **Fit View** | V, F (two-key) | R (single-key) |
| **Views** | V, I / V, T / V, 1 / V, 3 | R only |
| **Add Point** | P (tool mode) | Button click |
| **Generate** | G | Button click |
| **Toggle Domain** | D | D |
| **Open** | Ctrl+O | Ctrl+O |
| **Save** | Ctrl+S | Ctrl+S |

### Workflow

#### CAD Style Workflow
```
1. Click tool in toolbar (or press shortcut)
2. Tool becomes active (button highlighted)
3. Use tool in viewport (click, drag, etc.)
4. Tool stays active until deselected

Example: Add Point
→ Click ● in toolbar (or press P)
→ Button turns green (active state)
→ Click on domain multiple times
→ Each click adds a point
→ Click button again to deactivate
```

#### Modern Style Workflow
```
1. Click button in panel
2. Action executes immediately
3. Return to normal state

Example: Add Point
→ Click "• Pick Point" button
→ Button turns green temporarily
→ Click once on domain
→ Point added, mode exits
→ Must click button again for next point
```

### Use Cases

#### When to Use CAD Style ✓
- Working with CAD/CAM software daily
- Multi-monitor setup available
- Need to organize complex scenes
- Long work sessions (dark theme easier on eyes)
- Professional client presentations
- Integration with fabrication pipeline
- Team familiar with FreeCAD/SolidWorks
- Need customizable workspace

#### When to Use Modern Style ✓
- New to CAD software
- Simple, quick tasks
- Single-monitor laptop setup
- Teaching/demonstrations
- Prefer light themes
- Mobile/tablet-like interface
- Occasional use
- Consumer-facing applications

## Feature Matrix

| Feature | CAD | Modern |
|---------|-----|--------|
| Dark theme | ✓ | ✗ |
| Light theme | ✗ | ✓ |
| Dockable panels | ✓ | ✗ |
| Toolbars | ✓ | ✗ |
| Model tree | ✓ | ✗ |
| Object visibility toggles | ✓ | ✗ |
| Multi-section status bar | ✓ | ✗ |
| CAD shortcuts (V,F etc) | ✓ | ✗ |
| Tool-based workflow | ✓ | ✗ |
| Workspace customization | ✓ | ✗ |
| Floating panels | ✓ | ✗ |
| Branded header | ✗ | ✓ |
| Colorful icons | ✗ | ✓ |
| Material Design | ✗ | ✓ |
| Simple layout | ✗ | ✓ |
| One-click actions | ✗ | ✓ |
| Fixed panels | ✗ | ✓ |
| Tooltips | ✓ | ✓ |
| Keyboard shortcuts | ✓ | ✓ |
| 3D viewport | ✓ | ✓ |
| Generation parameters | ✓ | ✓ |
| Point selection | ✓ | ✓ |
| Save/Load | ✓ | ✓ |

## Choosing Your Style

### Quick Decision Guide

```
Do you use CAD software regularly?
├─ Yes → Use CAD Style
└─ No → Answer next question

Do you need to organize complex scenes?
├─ Yes → Use CAD Style
└─ No → Answer next question

Do you prefer dark themes?
├─ Yes → Use CAD Style
└─ No → Answer next question

Do you want a simple, fixed layout?
├─ Yes → Use Modern Style
└─ No → Use CAD Style

Still unsure? → Try CAD Style (it's the default)
```

### Personality Types

**CAD Style is for "Engineers"**:
- Analytical
- Detail-oriented
- Multi-tasker
- Customization enthusiast
- Efficiency-focused
- Professional workflows

**Modern Style is for "Creatives"**:
- Visual
- Intuitive
- Single-task focused
- Simplicity lover
- Aesthetics-conscious
- Casual users

## Code Examples

### Launch CAD Style
```python
from svv.visualize.gui import launch_gui

# Method 1: Default (CAD is now default)
launch_gui()

# Method 2: Explicit
launch_gui(style='cad')

# Method 3: With domain
from svv.domain import Domain
domain = Domain.load('my_domain.dmn')
launch_gui(domain=domain, style='cad')
```

### Launch Modern Style
```python
from svv.visualize.gui import launch_gui

# Explicit style parameter
launch_gui(style='modern')

# With domain
from svv.domain import Domain
domain = Domain.load('my_domain.dmn')
launch_gui(domain=domain, style='modern')
```

### Try Both
```python
from svv.visualize.gui import launch_gui

# Try CAD first
print("Launching CAD style...")
launch_gui(style='cad')

# Then try Modern
print("Launching Modern style...")
launch_gui(style='modern')

# Choose your favorite!
```

## Migration Path

### From Original GUI
```python
# Old code (pre-improvements):
from svv.visualize.gui import launch_gui
launch_gui()  # Had basic appearance

# New code (after improvements):
launch_gui()  # Now uses CAD style by default

# Or explicitly choose:
launch_gui(style='cad')     # Professional CAD interface
launch_gui(style='modern')  # Improved modern interface
```

### Updating Existing Scripts
```python
# Scripts with no style parameter will use CAD (new default)
launch_gui()  # CAD style

# To keep modern appearance, add parameter:
launch_gui(style='modern')

# Both work identically, just different appearance
```

## FAQ

### Q: Which style should I use?
**A**: CAD style if you're a fabrication professional or engineer. Modern style if you prefer light themes and simple layouts.

### Q: Can I switch styles without restarting?
**A**: No, style is set at launch. But you can close and relaunch with different style.

### Q: Do both styles have the same features?
**A**: Yes! Same generation engine, same parameters, same results. Only the interface differs.

### Q: Will my saved configurations work with both?
**A**: Yes! Configuration files (.json) work with both styles.

### Q: Which style is faster?
**A**: Identical performance. The theme is just CSS-like styling.

### Q: Can I create my own style?
**A**: Yes! Copy `cad_styles.py` or `styles.py` and modify the colors/styling.

### Q: What's the default now?
**A**: CAD style is the new default, as requested for fabrication users.

### Q: How do I go back to the "old" improved GUI?
**A**: Use `launch_gui(style='modern')` - that's the improved interface from before CAD style was added.

## Keyboard Shortcut Comparison

### CAD Style Shortcuts
```
File Operations:
  Ctrl+O ........ Open
  Ctrl+S ........ Save
  Ctrl+Q ........ Quit

View (Two-Key):
  V, F .......... Fit All
  V, I .......... Isometric
  V, T .......... Top
  V, 1 .......... Front
  V, 3 .......... Right

Tools:
  P ............. Add Point (toggle tool)
  G ............. Generate
  D ............. Toggle Domain
```

### Modern Style Shortcuts
```
File Operations:
  Ctrl+O ........ Open
  Ctrl+S ........ Save
  Ctrl+Q ........ Quit

View:
  R ............. Reset Camera
  D ............. Toggle Domain

(No tool shortcuts - use buttons)
```

## Summary Table

|  | CAD Style | Modern Style |
|---|-----------|--------------|
| **Best For** | Engineers, fabricators | General users |
| **Theme** | Dark | Light |
| **Complexity** | Higher (more features) | Lower (simpler) |
| **Learning Curve** | Steeper (many shortcuts) | Gentler |
| **Customization** | Extensive | Minimal |
| **Professional Use** | Excellent | Good |
| **Casual Use** | Good | Excellent |
| **Multi-Monitor** | Excellent | Good |
| **Long Sessions** | Excellent (dark theme) | Good |
| **Integration** | CAD/CAM workflows | General applications |
| **First Launch** | May seem complex | Immediately clear |
| **After Learning** | Very efficient | Simple and direct |

## Recommendation

### For Your Team
Based on your requirement for fabrication users who need a CAD-like environment:

**Primary Interface**: CAD Style ✓
- Default for all users
- Matches their existing tools
- Professional appearance
- Familiar shortcuts
- Dockable panels for their workflow

**Secondary Interface**: Modern Style
- Available for special cases
- Demos to non-technical clients
- Quick prototyping sessions
- Users who prefer light themes

### Implementation
```python
# In your workflow scripts
from svv.visualize.gui import launch_gui

# Default launch uses CAD style
launch_gui()  # Perfect for fabrication users

# Modern style available when needed
launch_gui(style='modern')
```

## Final Comparison Image

```
CAD Style                           Modern Style
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
█████████████████ Dark           ░░░░░░░░░░░░░░░░ Light
[▓▓▓][▓▓▓][▓▓▓] Toolbars         🌳 Header Card

▓│░░░░░░░░│▓ Dockable            ░│░░░░░░░░│░ Fixed
▓│░░░░░░░░│▓ Model Tree          ░│░░░░░░░░│░ Splitter
▓│░░░░░░░░│▓ + Properties        ░│░░░░░░░░│░ Panels
▓│░░░░░░░░│▓                     ░│░░░░░░░░│░

[▓▓][▓▓][▓▓][▓▓] Status          [░░░░░░░░░] Status

✓ Engineers                      ✓ General Users
✓ Fabrication                    ✓ Demos
✓ Multi-monitor                  ✓ Simplicity
✓ Long sessions                  ✓ Light theme
✓ Customizable                   ✓ Quick tasks
```

Both styles are production-ready and professionally designed. Choose based on your users and workflow!
