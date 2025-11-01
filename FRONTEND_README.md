# Frontend Integration - Classification Tags

This document explains how the classification tags are integrated into the React frontend for viewing and filtering training results.

## Overview

The React app now displays classification tags for each environment and provides filtering capabilities based on:
- **Convergence**: Whether training converged (yes/no, early/late)
- **Reshaped Return**: Training performance (high/low)
- **Evaluation Return**: Test performance (high/low)

## Features

### 1. Search Bar
- Type to filter environments by name
- Case-insensitive search
- Real-time filtering

### 2. Filter Controls
Four filter options available:
- **Convergence**: All / Converged (any) / Converged (early) / Converged (late) / Not Converged
- **Reshaped Return**: All / High / Low
- **Eval Return**: All / High / Low
- **Catalog Only**: Checkbox to show only catalog environments

### 3. Tag Display
Each environment shows colored tags:
- **Green tags**: Convergence status
  - Light green: `yes(early)`
  - Yellow: `yes(late)`
  - Red: `no`
- **Blue tags**: Reshaped return (R:high / R:low)
- **Purple/Orange tags**: Eval return (E:high / E:low)

### 4. Current Environment Info
When viewing a PDF, the top section displays:
- Environment name
- Full classification tags
- Color-coded badges

## Files Modified

### React App
- `front/my-react-app/src/App.tsx` - Main component with search/filter logic
- `front/my-react-app/src/App.css` - Styling for tags and filters
- `front/my-react-app/src/environment_classifications.json` - Classification data

### Scripts
- `update_frontend_classifications.sh` - Updates frontend with latest classifications

## Usage

### Running the Development Server

```bash
cd front/my-react-app
npm start
```

The app will open at http://localhost:3000

### Building for Production

```bash
cd front/my-react-app
npm run build
```

Serve the build:
```bash
npx serve -s build
```

### Updating Classifications

After running new training or evaluation:

```bash
# Option 1: Use the update script (recommended)
./update_frontend_classifications.sh

# Option 2: Manual steps
python3 analyze_training_simple.py --output environment_classifications.json
cp environment_classifications.json front/my-react-app/src/
cd front/my-react-app
npm run build
```

## Tag Color Scheme

### Convergence Tags
```css
yes(early) → Dark Green (#1b5e20) on Light Green (#c8e6c9)
yes(late)  → Orange (#f57f17) on Yellow (#fff9c4)
no         → Dark Red (#c62828) on Light Red (#ffcdd2)
```

### Reshaped Return Tags
```css
high → Dark Blue (#01579b) on Light Blue (#e1f5fe)
low  → Dark Pink (#880e4f) on Light Pink (#fce4ec)
```

### Eval Return Tags
```css
high → Dark Purple (#4a148c) on Light Purple (#f3e5f5)
low  → Dark Orange (#e65100) on Light Orange (#fff3e0)
```

## Data Flow

1. **Training/Evaluation** → Logs stored in `storage/{env_name}/`
2. **Analysis Script** → Generates `environment_classifications.json`
3. **React App** → Loads classifications and displays tags
4. **User Interaction** → Filters/searches through environments

## Example Workflows

### Workflow 1: Find High-Performing Environments

1. Set **Convergence** filter to "Converged (any)"
2. Set **Reshaped Return** filter to "High"
3. Set **Eval Return** filter to "High"
4. Browse filtered list

### Workflow 2: Debug Non-Converging Environments

1. Set **Convergence** filter to "Not Converged"
2. Review their training curves (PDFs)
3. Identify patterns

### Workflow 3: Compare Catalog Environments

1. Check "Catalog only" checkbox
2. Use search to filter by specific environment types
3. Compare tags and training curves side-by-side

## JSON Data Structure

The `environment_classifications.json` file contains:

```json
{
  "metadata": {
    "total_environments": 30,
    "converged": 26,
    "evaluated": 14,
    "thresholds": { ... }
  },
  "environments": {
    "DoorKey": {
      "env_name": "DoorKey",
      "convergence": "yes(early)",
      "reshaped_return": "low",
      "eval_return": "high",
      "total_frames": 100352,
      "total_updates": 49,
      "final_rreturn_mean": 0.1226,
      "eval_mean_return": 0.9324,
      "eval_std_return": 0.0234,
      "eval_episodes": 102
    },
    ...
  },
  "tags": {
    "DoorKey": "convergence:yes(early),reshaped_return:low,eval_return:high",
    ...
  }
}
```

## Customization

### Adjusting Classification Thresholds

Edit the analysis script parameters:

```bash
python3 analyze_training_simple.py \
  --convergence-threshold 0.03 \
  --high-reshaped-return 0.8 \
  --high-eval-return 0.95 \
  --output environment_classifications.json
```

### Modifying Tag Colors

Edit `front/my-react-app/src/App.css`:

```css
/* Change convergence tag colors */
.tag-convergence.tag-yes-early- {
  background-color: #your-bg-color;
  color: #your-text-color;
}
```

### Adding New Filters

1. Add state in `App.tsx`:
   ```typescript
   const [newFilter, setNewFilter] = useState<string>('all');
   ```

2. Add filter UI:
   ```tsx
   <select value={newFilter} onChange={(e) => setNewFilter(e.target.value)}>
     <option value="all">All</option>
     ...
   </select>
   ```

3. Update filter logic in `filteredEnvironments` useMemo

## Troubleshooting

### Tags Not Showing
- Ensure `environment_classifications.json` is in `src/` directory
- Check that environment names match between file-list and classifications
- Rebuild the app: `npm run build`

### Filters Not Working
- Clear browser cache
- Check console for JavaScript errors
- Verify classification data structure

### Build Errors
- Run `npm install` to ensure dependencies are installed
- Check TypeScript errors: `npm run build`
- Verify all imports are correct

## Performance Considerations

- Classifications are loaded once at app startup (useMemo)
- Filtering is done client-side for instant results
- For 100+ environments, consider pagination
- Large PDFs may take time to load

## Future Enhancements

Potential improvements:
- [ ] Sort environments by tags
- [ ] Export filtered results
- [ ] Multi-environment comparison view
- [ ] Real-time classification updates
- [ ] Custom tag creation
- [ ] Advanced search (regex, multiple criteria)
- [ ] Bookmarking favorite environments
- [ ] Notes/comments on environments

## Support

For issues or questions:
1. Check console for errors
2. Verify data files exist and are valid JSON
3. Ensure all scripts have execute permissions
4. Review this README and ANALYSIS_README.md
