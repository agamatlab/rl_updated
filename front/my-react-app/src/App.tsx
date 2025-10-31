import React, {useState, useMemo} from 'react';
import './App.css';
import PDFViewer from './components/PDFViewer';
import files from './file-list.json';
import classifications from './environment_classifications.json';

const STORAGE_BASE =
  (process.env.REACT_APP_STORAGE_BASE_URL ?? '/storage').replace(/\/+$/, '');

interface EnvironmentInfo {
  name: string;
  tags: string;
  convergence?: string;
  reshaped_return?: string;
  eval_return?: string;
}

function App() {
  const [file, setFile] = useState<string>(files[3]);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [filterConvergence, setFilterConvergence] = useState<string>('all');
  const [filterReshapedReturn, setFilterReshapedReturn] = useState<string>('all');
  const [filterEvalReturn, setFilterEvalReturn] = useState<string>('all');
  const [showOnlyCatalog, setShowOnlyCatalog] = useState<boolean>(false);

  const pdfPath = `${STORAGE_BASE}/${file}/training_curves.pdf`;

  // Prepare environment data with tags
  const environments: EnvironmentInfo[] = useMemo(() => {
    return files.map(fileName => {
      const tags = (classifications.tags as any)[fileName] || '';
      const envData = (classifications.environments as any)[fileName];

      return {
        name: fileName,
        tags: tags,
        convergence: envData?.convergence,
        reshaped_return: envData?.reshaped_return,
        eval_return: envData?.eval_return
      };
    });
  }, []);

  // Filter environments
  const filteredEnvironments = useMemo(() => {
    return environments.filter(env => {
      // Search query filter
      if (searchQuery && !env.name.toLowerCase().includes(searchQuery.toLowerCase())) {
        return false;
      }

      // Catalog filter
      if (showOnlyCatalog && !env.name.startsWith('catalog')) {
        return false;
      }

      // Convergence filter
      if (filterConvergence !== 'all') {
        if (filterConvergence === 'yes' && !env.convergence?.startsWith('yes')) {
          return false;
        }
        if (filterConvergence === 'yes(early)' && env.convergence !== 'yes(early)') {
          return false;
        }
        if (filterConvergence === 'yes(late)' && env.convergence !== 'yes(late)') {
          return false;
        }
        if (filterConvergence === 'no' && env.convergence !== 'no') {
          return false;
        }
      }

      // Reshaped return filter
      if (filterReshapedReturn !== 'all' && env.reshaped_return !== filterReshapedReturn) {
        return false;
      }

      // Eval return filter
      if (filterEvalReturn !== 'all') {
        if (!env.eval_return || env.eval_return !== filterEvalReturn) {
          return false;
        }
      }

      return true;
    });
  }, [environments, searchQuery, filterConvergence, filterReshapedReturn, filterEvalReturn, showOnlyCatalog]);

  // Get current environment info
  const currentEnvInfo = environments.find(env => env.name === file);

  return (
    <div className="App">
      <header className="App-header">
        <h1>RL Environment Training Viewer</h1>
        <p className="subtitle">Total Environments: {filteredEnvironments.length} / {environments.length}</p>
      </header>
      <main className="App-main">
        <div className="file-list-container">
          <h2>Environments</h2>

          {/* Search Bar */}
          <div className="search-container">
            <input
              type="text"
              placeholder="Search environments..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="search-input"
            />
          </div>

          {/* Filters */}
          <div className="filters-container">
            <div className="filter-group">
              <label>Convergence:</label>
              <select value={filterConvergence} onChange={(e) => setFilterConvergence(e.target.value)}>
                <option value="all">All</option>
                <option value="yes">Converged (any)</option>
                <option value="yes(early)">Converged (early)</option>
                <option value="yes(late)">Converged (late)</option>
                <option value="no">Not Converged</option>
              </select>
            </div>

            <div className="filter-group">
              <label>Reshaped Return:</label>
              <select value={filterReshapedReturn} onChange={(e) => setFilterReshapedReturn(e.target.value)}>
                <option value="all">All</option>
                <option value="high">High</option>
                <option value="low">Low</option>
              </select>
            </div>

            <div className="filter-group">
              <label>Eval Return:</label>
              <select value={filterEvalReturn} onChange={(e) => setFilterEvalReturn(e.target.value)}>
                <option value="all">All</option>
                <option value="high">High</option>
                <option value="low">Low</option>
              </select>
            </div>

            <div className="filter-group checkbox-group">
              <label>
                <input
                  type="checkbox"
                  checked={showOnlyCatalog}
                  onChange={(e) => setShowOnlyCatalog(e.target.checked)}
                />
                Catalog only
              </label>
            </div>
          </div>

          {/* Environment List */}
          <ul className="file-list">
            {filteredEnvironments.map((env, index) => (
              <li
                key={index}
                onClick={() => setFile(env.name)}
                className={file === env.name ? 'active' : ''}
              >
                <div className="env-name">{env.name}</div>
                {env.tags && (
                  <div className="env-tags">
                    {env.convergence && (
                      <span className={`tag tag-convergence tag-${env.convergence.replace(/[()]/g, '-')}`}>
                        {env.convergence}
                      </span>
                    )}
                    {env.reshaped_return && (
                      <span className={`tag tag-reshaped tag-${env.reshaped_return}`}>
                        R:{env.reshaped_return}
                      </span>
                    )}
                    {env.eval_return && (
                      <span className={`tag tag-eval tag-${env.eval_return}`}>
                        E:{env.eval_return}
                      </span>
                    )}
                  </div>
                )}
              </li>
            ))}
          </ul>
        </div>
        <div className="pdf-viewer-container">
          {currentEnvInfo && (
            <div className="current-env-info">
              <h3>{currentEnvInfo.name}</h3>
              {currentEnvInfo.tags && (
                <div className="current-env-tags">
                  {currentEnvInfo.convergence && (
                    <span className={`tag tag-convergence tag-${currentEnvInfo.convergence.replace(/[()]/g, '-')}`}>
                      Convergence: {currentEnvInfo.convergence}
                    </span>
                  )}
                  {currentEnvInfo.reshaped_return && (
                    <span className={`tag tag-reshaped tag-${currentEnvInfo.reshaped_return}`}>
                      Reshaped Return: {currentEnvInfo.reshaped_return}
                    </span>
                  )}
                  {currentEnvInfo.eval_return && (
                    <span className={`tag tag-eval tag-${currentEnvInfo.eval_return}`}>
                      Eval Return: {currentEnvInfo.eval_return}
                    </span>
                  )}
                </div>
              )}
            </div>
          )}
          <PDFViewer file={pdfPath} />
        </div>
      </main>
    </div>
  );
}

export default App;
