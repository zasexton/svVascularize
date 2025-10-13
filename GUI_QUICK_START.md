# svVascularize GUI - Quick Start Guide

## Launch the GUI

### Method 1: Direct Launch
```bash
python -m svv.visualize.gui
```

### Method 2: Safe Launcher (Recommended)
```bash
python launch_gui_safe.py
```

### Method 3: From Python Script
```python
from svv.visualize.gui import launch_gui
launch_gui()
```

## Your First Tree in 5 Steps

### Step 1: Load a Domain
1. Click **📂 Load Domain** in the menu (or press `Ctrl+O`)
2. Select your `.dmn` domain file
3. Watch the 3D visualization appear
4. Status bar shows: "✔ Domain loaded - Ready to configure trees"

### Step 2: Choose Mode
In the Parameter Panel (right side):
- Select **🌳 Single Tree** from the dropdown

### Step 3: Set Parameters
Adjust these values:
- **Number of Vessels**: `100` (start small for testing)
- **Physical Clearance**: `0.0` (vessels can touch)
- **Convexity Tolerance**: `0.01` (default is fine)

### Step 4: Select Start Point
1. Click **• Pick Point (Click on Domain)** button (turns green)
2. Click anywhere on the blue domain mesh in the 3D view
3. A red sphere appears at your click location
4. Point is added to the list

### Step 5: Generate!
1. Click the big orange **▶ Generate Tree/Forest** button
2. Watch the progress bar
3. See your tree appear in the 3D view!
4. Success dialog shows statistics

**Congratulations!** You've generated your first vascular tree.

## Keyboard Shortcuts Cheat Sheet

| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Load domain file |
| `Ctrl+S` | Save configuration |
| `Ctrl+Q` | Quit application |
| `R` | Reset camera view |
| `D` | Toggle domain visibility |

## Common Workflows

### Workflow 1: Single Tree with Custom Direction

```
1. Load domain (Ctrl+O)
2. Select "Single Tree" mode
3. Pick a start point
4. Select the point in the list
5. Check "→ Use Custom Direction"
6. Set direction: X=0, Y=0, Z=1 (grows upward)
7. Click "Normalize Direction"
8. Click "▶ Generate Tree/Forest"
9. Done!
```

### Workflow 2: Forest with Two Networks

```
1. Load domain (Ctrl+O)
2. Select "🌲 Forest (Multiple Trees)" mode
3. Set Networks to 2
4. Select Network 0, Tree 0
5. Pick first start point
6. Select Network 0, Tree 1
7. Pick second start point
8. Select Network 1, Tree 0
9. Pick third start point
10. Select Network 1, Tree 1
11. Pick fourth start point
12. Enable Competition (check box)
13. Click "▶ Generate Tree/Forest"
14. See multi-colored tree networks!
```

### Workflow 3: Save and Export

```
1. Configure parameters as desired
2. Add start points
3. Click "💾 Export Configuration" (Ctrl+S)
4. Choose filename (e.g., "my_config.json")
5. Click Save
6. Status: "✔ Configuration saved successfully"
7. Reuse anytime: Load domain → Generate with saved settings
```

## Understanding the Interface

### Left Side: 3D Visualization
- **Blue mesh**: Your domain
- **Red spheres**: Start points you've added
- **Blue arrows**: Custom directions (if set)
- **Colored cylinders**: Generated vessels
- **Mouse controls**:
  - Left drag: Rotate view
  - Right drag: Pan view
  - Scroll: Zoom in/out

### Right Side: Control Panels

#### Top Panel: Start Points & Directions
- Configure where trees start growing
- Set custom growth directions
- Manage multiple networks and trees

#### Bottom Panel: Parameters
- Choose single tree or forest
- Set vessel count and spacing
- Configure competition and randomness
- Generate and export

### Bottom: Status Bar
- Real-time feedback on all operations
- Success/error messages with icons
- Always check here for current status

## Tips for Success

### 🎯 Start Simple
- Begin with 100-200 vessels
- Use single tree mode first
- Add complexity gradually

### 🎨 Visual Feedback
- Green button = picking mode active
- Orange button = main action
- Red button = destructive (be careful!)
- Icons show what everything does

### 💡 Use Tooltips
- Hover over ANY control for help
- Explanations for all parameters
- Learn as you go

### 📊 Read Status Messages
- Status bar shows real-time updates
- Success dialogs include statistics
- Error messages suggest solutions

### ⌨️ Learn Shortcuts
- `Ctrl+O`: Load faster
- `Ctrl+S`: Save frequently
- `R`: Reset view when lost
- `D`: Hide domain to see trees better

### 🔄 Iterate Quickly
1. Generate with low vessel count
2. Check if parameters work
3. Adjust as needed
4. Generate again with higher count

## Parameter Guide

### Number of Vessels
- **Range**: 1-100,000
- **Typical**: 100-1,000
- **Tip**: Start small, increase gradually

### Physical Clearance
- **Range**: 0.0-10.0
- **Default**: 0.0
- **Meaning**: Minimum gap between vessels
- **Tip**: 0.0 allows touching, >0 adds space

### Convexity Tolerance
- **Range**: 0.0-1.0
- **Default**: 0.01
- **Meaning**: How strictly to check domain bounds
- **Tip**: Lower = stricter checking

### Competition (Forest Only)
- **Effect**: Trees compete for space
- **Tip**: Enable for more realistic forests

### Decay Probability (Forest Only)
- **Range**: 0.0-1.0
- **Default**: 0.9
- **Meaning**: How likely vessels decay in competition
- **Tip**: Higher = more decay

### Random Seed
- **Default**: Random (changes each time)
- **Use**: Set a number for reproducible results
- **Tip**: -1 = random, ≥0 = specific seed

## Troubleshooting Quick Fixes

### Problem: Can't see the domain
**Fix**: Press `D` to toggle visibility, or click View → Toggle Domain Visibility

### Problem: Camera view is wrong
**Fix**: Press `R` to reset camera, or click View → Reset Camera

### Problem: Pick point button doesn't work
**Fix**: Make sure domain is loaded first. Status should say "Domain loaded"

### Problem: Generate button does nothing
**Fix**: Check status bar for error. Ensure domain is loaded and at least one point is added (for single tree)

### Problem: Tree generation is slow
**Fix**: This is normal for high vessel counts. Use progress dialog's cancel button if needed

### Problem: Can't see generated tree
**Fix**:
1. Press `D` to hide domain
2. Press `R` to reset camera
3. Check if generation succeeded (status bar)

## What to Do Next

### Experiment!
- Try different vessel counts
- Test various start positions
- Compare single trees vs forests
- Enable/disable competition

### Save Your Work
- Export configurations you like
- Keep a library of useful settings
- Share configs with team members

### Learn Advanced Features
- Custom directions for angled growth
- Multiple networks for complex scenarios
- Competition for realistic forests
- Parameter optimization for specific goals

### Get Help
- Hover over controls for tooltips
- Check status bar for messages
- Read error dialogs carefully
- See GUI_IMPROVEMENTS.md for details

## Example Use Cases

### Use Case 1: Vascular Network for Heart
```
Mode: Forest
Networks: 2 (arterial + venous)
Vessels: 500 per network
Competition: Enabled
Start points: At major vessels
```

### Use Case 2: Brain Vasculature
```
Mode: Forest
Networks: 1
Trees: 4 (different entry points)
Vessels: 1000 total
Competition: Enabled
Directions: Inward from surface
```

### Use Case 3: Simple Test Case
```
Mode: Single Tree
Vessels: 50
Clearance: 0.1
Start: Center of domain
Direction: Automatic
```

## Learning Path

### Week 1: Basics
- [ ] Launch GUI successfully
- [ ] Load a domain
- [ ] Generate a simple tree
- [ ] Save a configuration

### Week 2: Intermediate
- [ ] Use custom directions
- [ ] Create a forest
- [ ] Adjust clearance
- [ ] Optimize parameters

### Week 3: Advanced
- [ ] Multi-network forests
- [ ] Competition settings
- [ ] Reproducible results (seeds)
- [ ] Export and reuse configs

## Best Practices

### Do's ✓
- ✓ Start with low vessel counts for testing
- ✓ Save successful configurations
- ✓ Read tooltip descriptions
- ✓ Check status bar regularly
- ✓ Use keyboard shortcuts
- ✓ Reset camera view when lost

### Don'ts ✗
- ✗ Don't start with 10,000 vessels
- ✗ Don't ignore error messages
- ✗ Don't forget to load domain first
- ✗ Don't skip parameter descriptions
- ✗ Don't clear all points accidentally

## Resources

- **GUI_IMPROVEMENTS.md**: Detailed list of all improvements
- **GUI_FEATURES.md**: Complete feature reference
- **GUI_UPGRADE_SUMMARY.md**: Before/after comparison

## Getting Support

If you encounter issues:
1. Check status bar for error messages
2. Read error dialog carefully
3. Verify domain file is valid
4. Try with default parameters
5. Restart GUI if needed
6. Check documentation files

## Summary

You now know how to:
- ✓ Launch the GUI
- ✓ Load domains
- ✓ Generate trees and forests
- ✓ Configure parameters
- ✓ Save configurations
- ✓ Use keyboard shortcuts
- ✓ Troubleshoot common issues

**Next Step**: Try generating your first tree following the 5-step guide above!

Happy vascularizing! 🌳
