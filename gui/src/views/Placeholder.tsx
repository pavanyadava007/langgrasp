import { Card } from "../components/ui";

/** A view whose phase has not been built yet. It says what will be here and where the data comes from, rather
 *  than showing an empty shell that looks broken. */
export function Placeholder({ title, phase, what, sources }: { title: string; phase: string; what: string; sources?: string[] }) {
  return (
    <Card title={title} subtitle={`Not built yet: ${phase}.`}>
      <p className="max-w-2xl text-sm">{what}</p>
      {sources && (
        <>
          <h3 className="mt-3 text-xs font-medium uppercase tracking-wide text-fg-muted">It will read</h3>
          <ul className="num mt-1 list-inside list-disc text-sm text-fg-muted">
            {sources.map((s) => (
              <li key={s}>{s}</li>
            ))}
          </ul>
        </>
      )}
    </Card>
  );
}
