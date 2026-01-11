import { Highlight, themes } from 'prism-react-renderer';
import './SyntaxHighlighter.css';

interface SyntaxHighlighterProps {
  code: string;
  language?: string;
}

export default function SyntaxHighlighter({ code, language = 'python' }: SyntaxHighlighterProps) {
  return (
    <Highlight theme={themes.oneDark} code={code} language={language}>
      {({ className, style, tokens, getLineProps, getTokenProps }) => (
        <pre className={`syntax-highlighter ${className}`} style={style}>
          {tokens.map((line, i) => (
            <div key={i} {...getLineProps({ line })} className="code-line">
              <span className="line-number">{i + 1}</span>
              <span className="line-content">
                {line.map((token, key) => (
                  <span key={key} {...getTokenProps({ token })} />
                ))}
              </span>
            </div>
          ))}
        </pre>
      )}
    </Highlight>
  );
}
