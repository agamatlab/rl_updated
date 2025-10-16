import React, {useState} from 'react';
import './App.css';
import PDFViewer from './components/PDFViewer';
import files from './file-list.json';

const STORAGE_BASE =
  (process.env.REACT_APP_STORAGE_BASE_URL ?? '/storage').replace(/\/+$/, '');

function App() {
  const [file, setFile] = useState<string>(files[3]);

  const pdfPath = `${STORAGE_BASE}/${file}/training_curves.pdf`;

  return (
    <div className="App">
      <header className="App-header">
        <h1>PDF Viewer</h1>
      </header>
      <main className="App-main">
        <div className="file-list-container">
          <h2>Available PDF Files:</h2>
          <ul className="file-list">
            {files.map((file, index) => (
              (file.startsWith("catalog")) ? <li key={index} onClick={() => setFile(file)}>{file}</li> : null
            ))}
          </ul>
        </div>
        <div className="pdf-viewer-container">
          <PDFViewer file={pdfPath} />
        </div>
      </main>
    </div>
  );
}

export default App;
