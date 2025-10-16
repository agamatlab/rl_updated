import { useEffect, useState } from 'react';

interface PDFViewerProps {
  file: string;
}

export default function PDFViewer({ file }: PDFViewerProps) {
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    let objectUrl: string | null = null;

    async function loadPdf() {
      setPdfUrl(null);
      setLoadError(null);

      try {
        const response = await fetch(file, {
          signal: controller.signal,
          headers: { Accept: 'application/pdf' },
        });

        if (!response.ok) {
          throw new Error(`Server responded with ${response.status}`);
        }

        const blob = await response.blob();
        if (blob.type && !blob.type.includes('pdf')) {
          throw new Error(`Unexpected content type: ${blob.type}`);
        }

        objectUrl = URL.createObjectURL(blob);
        setPdfUrl(objectUrl);
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }

        setLoadError(
          error instanceof Error ? error.message : 'Failed to load PDF.'
        );
      }
    }

    void loadPdf();

    return () => {
      controller.abort();
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [file]);

  return (
    <div className="pdf-iframe-wrapper">
      {pdfUrl ? (
        <iframe
          src={pdfUrl}
          title="PDF preview"
          className="pdf-frame"
          loading="lazy"
        />
      ) : (
        <div className="pdf-loading">
          {loadError ? `Unable to display PDF: ${loadError}` : 'Loading PDF...'}
        </div>
      )}
      <p className="pdf-download-hint">
        Having trouble loading?{' '}
        <a href={file} target="_blank" rel="noopener noreferrer">
          Open the PDF in a new tab
        </a>
        .
      </p>
    </div>
  );
}
