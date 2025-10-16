import React, { useState } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';

// Use the locally bundled worker instead of a CDN path to avoid 404s in production.
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdf.worker.min.js',
  import.meta.url,
).toString();

interface PDFViewerProps {
  file: string;
}

function PDFViewer({ file }: PDFViewerProps) {
  const [numPages, setNumPages] = useState<number | null>(null);
  const [pageNumber, setPageNumber] = useState<number>(1);

  function onDocumentLoadSuccess({ numPages }: { numPages: number }) {
    setNumPages(numPages);
    setPageNumber(1);
  }

  return (
    <div>
      <Document file={file} onLoadSuccess={onDocumentLoadSuccess}>
        <Page pageNumber={pageNumber} />
      </Document>
      <p>
        Page {pageNumber} of {numPages}
      </p>
    </div>
  );
}

export default PDFViewer;
