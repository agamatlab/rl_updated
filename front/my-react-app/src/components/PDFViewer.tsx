interface PDFViewerProps {
  file: string;
}

export default function PDFViewer({ file }: PDFViewerProps) {
  return (
    <div className="pdf-iframe-wrapper">
      <iframe
        src={file}
        title="PDF preview"
        className="pdf-frame"
        loading="lazy"
      />
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
