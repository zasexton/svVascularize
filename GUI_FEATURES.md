# svVascularize GUI Features Guide

## Quick Reference: What's New

### Visual Elements

#### 1. Branded Header
```
┌────────────────────────────────────────────────────┐
│ 🌳 svVascularize                                   │
│ Interactive vascular tree and forest generation   │
└────────────────────────────────────────────────────┘
```
- Blue background (#2196F3)
- Large title with tree icon
- Descriptive subtitle
- Professional appearance

#### 2. Menu Bar with Icons
```
File                View               Help
├─ 📂 Load Domain   ├─ 📷 Reset Camera ├─ ℹ About
├─ 💾 Save Config   └─ 👁 Toggle Domain
└─ Exit
```

**Keyboard Shortcuts:**
- `Ctrl+O` - Load Domain
- `Ctrl+S` - Save Configuration
- `Ctrl+Q` - Quit
- `R` - Reset Camera
- `D` - Toggle Domain

#### 3. Parameter Panel

**Generation Mode:**
```
┌─ ⚙ Generation Mode ────────────────┐
│ [🌳 Single Tree         ▼]         │
└────────────────────────────────────┘
```

**Tree Parameters:**
```
┌─ ⚙ Tree Parameters ────────────────┐
│ Number of Vessels:     [100    ]   │
│ Physical Clearance:    [0.0000 ]   │
│ Convexity Tolerance:   [0.010  ]   │
└────────────────────────────────────┘
```

**Forest Parameters** (when forest mode selected):
```
┌─ 🌲 Forest Parameters ─────────────┐
│ Competition:  [✓] Enable           │
│ Decay Probability:     [0.90   ]   │
└────────────────────────────────────┘
```

**Advanced Parameters:**
```
┌─ ⚙ Advanced Parameters ────────────┐
│ Random Seed:           [Random  ]   │
└────────────────────────────────────┘
```

**Action Buttons:**
```
┌──────────────────────────────────────┐
│ ▶ Generate Tree/Forest              │ <- Orange (Primary)
├──────────────────────────────────────┤
│ 💾 Export Configuration              │ <- Blue outline (Secondary)
└──────────────────────────────────────┘
```

#### 4. Point Selector

**Network Configuration:**
```
┌─ • Start Points & Directions ──────┐
│ Networks: [1 ]                      │
│ Network: [Network 0 ▼] Tree: [0 ▼] │
│                                      │
│ Points:                              │
│ ┌──────────────────────────────────┐│
│ │ P0: N0T0 (1.23, 2.45, 3.67)     ││
│ │ P1: N0T1 (4.56, 5.67, 6.78) [Dir]││
│ └──────────────────────────────────┘│
└──────────────────────────────────────┘
```

**Point Picking:**
```
┌────────────────────────────────────┐
│ • Pick Point (Click on Domain)    │ <- Blue, turns green when active
├────────────────────────────────────┤
│ + Manual Input...                  │ <- Blue outline
└────────────────────────────────────┘
```

**Point Details:**
```
┌─ Point Details ────────────────────┐
│ Position: (1.234, 2.456, 3.678)   │
│ → Use Custom Direction             │
│ Direction:                          │
│ X: [0.000] Y: [0.000] Z: [1.000]  │
│ [Normalize Direction]              │
└────────────────────────────────────┘
```

**Point Actions:**
```
┌────────────────┬───────────────────┐
│ ➖ Delete Selected │ ✖ Clear All    │ <- Both red (Danger)
└────────────────┴───────────────────┘
```

### Status Bar Messages

The status bar provides real-time feedback:

**Loading:**
```
📂 Loading domain...
```

**Success:**
```
✔ Domain loaded - Ready to configure trees
✔ Configuration saved successfully
✔ Tree generated successfully with 100 vessels
✔ Forest generated successfully with 500 vessels across 2 networks
```

**Working:**
```
▶ Generating tree with 100 vessels...
▶ Generating forest with 500 total vessels...
```

**Errors:**
```
⚠ Failed to load domain
⚠ Generation failed
```

### Dialog Messages

#### Success Dialogs
```
┌─ ✔ Success ───────────────────────────┐
│                                        │
│ Tree generated successfully!           │
│                                        │
│ Total vessels: 100                     │
│ Physical clearance: 0.0                │
│ Convexity tolerance: 0.01              │
│                                        │
│              [OK]                      │
└────────────────────────────────────────┘
```

#### Warning Dialogs
```
┌─ ⚠ No Domain ─────────────────────────┐
│                                        │
│ Please load a domain file before      │
│ generating trees.                      │
│                                        │
│ Use File > Load Domain to get started.│
│                                        │
│              [OK]                      │
└────────────────────────────────────────┘
```

#### Error Dialogs
```
┌─ ⚠ Error Loading Domain ──────────────┐
│                                        │
│ Failed to load domain file:            │
│                                        │
│ FileNotFoundError: 'file.dmn' not found│
│                                        │
│ Please ensure the file is a valid     │
│ domain file (.dmn)                     │
│                                        │
│              [OK]                      │
└────────────────────────────────────────┘
```

#### Progress Dialogs
```
┌─ 🌳 Generating tree... ───────────────┐
│                                        │
│ This may take a few moments.           │
│                                        │
│ ████████████░░░░░░░░ 60%              │
│                                        │
│            [✖ Cancel]                  │
└────────────────────────────────────────┘
```

## Tooltip Examples

Hover over any control to see helpful tooltips:

| Control | Tooltip |
|---------|---------|
| Number of Vessels | Total number of vessel segments to generate |
| Physical Clearance | Minimum distance between vessel walls (0 = allow touching) |
| Convexity Tolerance | Tolerance for domain convexity checking (smaller = stricter) |
| Enable Competition | Enable competition between trees for territory |
| Decay Probability | Probability of decay for competitive growth (0-1) |
| Random Seed | Set random seed for reproducible results (-1 = random) |
| Pick Point Button | Click on the 3D domain to select a start point |
| Manual Input Button | Enter point coordinates manually |
| Use Custom Direction | Specify a custom growth direction for this start point |
| Delete Selected | Delete the currently selected point |
| Clear All | Remove all start points |
| Generate Button | Generate vascular tree or forest with current parameters |
| Export Button | Export current configuration to JSON file |

## Color Scheme

### Primary Colors
- **Primary Blue**: #2196F3 - Main actions, headers
- **Primary Dark**: #1976D2 - Hover states
- **Primary Light**: #BBDEFB - Selections, backgrounds

### Accent Colors
- **Secondary Orange**: #FF5722 - Important actions
- **Success Green**: #4CAF50 - Success states
- **Warning Orange**: #FF9800 - Warnings
- **Error Red**: #F44336 - Errors, destructive actions

### Neutral Colors
- **Background**: #FAFAFA - Main background
- **Surface**: #FFFFFF - Card/panel backgrounds
- **Text Primary**: #212121 - Main text
- **Text Secondary**: #757575 - Secondary text
- **Divider**: #E0E0E0 - Borders, dividers

## Button Types

### Primary Button (Orange)
```
┌──────────────────────────────────┐
│ ▶ Generate Tree/Forest          │
└──────────────────────────────────┘
```
- Use for: Main actions, primary workflows
- Color: Orange (#FF5722)
- Style: Bold, prominent

### Secondary Button (Blue Outline)
```
┌──────────────────────────────────┐
│ 💾 Export Configuration          │
└──────────────────────────────────┘
```
- Use for: Supporting actions, alternatives
- Color: Blue outline (#2196F3)
- Style: Outlined, less prominent

### Danger Button (Red)
```
┌──────────────────────────────────┐
│ ✖ Clear All                      │
└──────────────────────────────────┘
```
- Use for: Destructive actions, deletions
- Color: Red (#F44336)
- Style: Bold, warning color

### Checkable Button (Changes Color)
```
Normal:  │ • Pick Point (Click on Domain)    │
Active:  │ • Picking... (Click Domain)       │ (Green)
```
- Use for: Toggle states, modes
- Color: Blue → Green when active
- Style: State-dependent

## Icon Reference

| Icon | Unicode | Usage |
|------|---------|-------|
| 🌳 | U+1F333 | Trees, main branding |
| 🌲 | U+1F332 | Forests |
| 📂 | U+1F4C2 | Open/Load |
| 💾 | U+1F4BE | Save |
| ⚙ | U+2699 | Settings, parameters |
| • | U+2022 | Points |
| → | U+2192 | Arrows, directions |
| + | U+002B | Add |
| ➖ | U+2796 | Remove |
| ✔ | U+2714 | Success |
| ✖ | U+2716 | Cancel, error |
| ℹ | U+2139 | Information |
| ⚠ | U+26A0 | Warning |
| ▶ | U+25B6 | Play, start |
| 📷 | U+1F4F7 | Camera |
| 👁 | U+1F441 | View, visibility |

## Responsive Design

The GUI adapts to different window sizes:

### Default Size (1400x900)
- Optimal viewing experience
- Splitter: 70% visualization, 30% controls

### Minimum Size (800x600)
- Scroll areas activate in parameter panels
- Controls remain fully functional
- Text remains readable

### Large Screens (>1920px)
- More visualization space
- Same proportions maintained
- Text scales appropriately

## Accessibility Features

1. **High Contrast**: All text meets WCAG AA standards
2. **Clear Focus**: Visible focus indicators on all controls
3. **Keyboard Navigation**: Tab through all controls
4. **Tooltips**: Context for screen readers
5. **Status Updates**: Announced to screen readers
6. **Large Click Targets**: Buttons are min 28px height
7. **Color Independence**: Icons supplement color meaning

## Usage Examples

### Example 1: Generate a Single Tree
1. Click "📂 Load Domain" or press `Ctrl+O`
2. Select your domain file
3. Status: "✔ Domain loaded - Ready to configure trees"
4. Set "Number of Vessels" to desired value
5. Click "• Pick Point" (turns green)
6. Click on domain to select start point
7. Click "▶ Generate Tree/Forest"
8. Watch progress dialog
9. Status: "✔ Tree generated successfully with X vessels"

### Example 2: Generate a Forest with Custom Directions
1. Load domain (as above)
2. Select "🌲 Forest (Multiple Trees)" from mode dropdown
3. Set number of networks
4. For each network/tree:
   - Pick a start point
   - Check "→ Use Custom Direction"
   - Set X, Y, Z values
   - Click "Normalize Direction"
5. Configure forest parameters (competition, decay)
6. Click "▶ Generate Tree/Forest"
7. Status: "✔ Forest generated successfully..."

### Example 3: Save and Reuse Configuration
1. Configure parameters and points as desired
2. Click "💾 Export Configuration" or press `Ctrl+S`
3. Choose save location
4. Success dialog appears
5. Status: "✔ Configuration saved successfully"
6. Reuse: Load domain, import config, generate

## Tips for Best Results

1. **Use Tooltips**: Hover over any control to learn what it does
2. **Check Status Bar**: Watch for feedback on all actions
3. **Normalize Directions**: Use the normalize button for unit vectors
4. **Save Often**: Export configurations to preserve your work
5. **Use Shortcuts**: Learn keyboard shortcuts for faster workflow
6. **Read Dialogs**: Success messages include helpful statistics
7. **Experiment**: Adjust parameters and see results immediately

## Troubleshooting

### Issue: GUI looks plain/unstyled
**Solution**: Theme is applied automatically. If not visible, ensure PySide6 is up to date.

### Issue: Icons don't display
**Solution**: Unicode icons require proper font support. Most systems support them by default.

### Issue: Tooltips don't appear
**Solution**: Hover longer (1-2 seconds). Check Qt settings for tooltip delays.

### Issue: Status messages disappear quickly
**Solution**: They're visible for 5 seconds. Check again after operations complete.

### Issue: Progress dialog doesn't show
**Solution**: For fast operations (<1 second), dialog may not appear. This is normal.

## Summary of Benefits

✅ **Professional Appearance**: Modern, polished design
✅ **Easy to Learn**: Tooltips and clear labels guide users
✅ **Fast Workflow**: Keyboard shortcuts save time
✅ **Better Feedback**: Know exactly what's happening
✅ **Fewer Errors**: Clear validation and helpful messages
✅ **More Confidence**: Detailed success confirmations
✅ **Accessible**: Works for all users
✅ **Consistent**: Same patterns throughout
✅ **Maintainable**: Clean, organized code
✅ **Future-Ready**: Easy to extend and customize

The svVascularize GUI is now a pleasure to use and ready for production deployment!
