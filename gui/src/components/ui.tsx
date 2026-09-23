import { useId, useState, type ReactNode } from "react";
import { glossaryFor } from "../lib/glossary";

export function Button({
  children,
  onClick,
  variant = "subtle",
  disabled,
  title,
  full,
  type = "button",
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "primary" | "subtle" | "danger" | "ghost";
  disabled?: boolean;
  title?: string;
  full?: boolean;
  type?: "button" | "submit";
}) {
  const base = "inline-flex items-center justify-center gap-2 rounded-m px-3 py-2 text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed";
  const variants = {
    primary: "bg-accent text-[color:var(--accent-fg)] hover:brightness-110",
    subtle: "bg-surface-2 text-fg border border-edge hover:border-accent",
    danger: "bg-estop text-white hover:brightness-110",
    ghost: "text-fg-muted hover:text-fg",
  } as const;
  return (
    <button type={type} onClick={onClick} disabled={disabled} title={title} className={`${base} ${variants[variant]} ${full ? "w-full" : ""}`}>
      {children}
    </button>
  );
}

export function Card({ title, subtitle, children, right, footer }: { title?: ReactNode; subtitle?: ReactNode; children: ReactNode; right?: ReactNode; footer?: ReactNode }) {
  return (
    <section className="rounded-l border border-edge bg-surface" style={{ boxShadow: "var(--shadow)" }}>
      {(title || right) && (
        <header className="flex items-start justify-between gap-3 border-b border-edge px-4 py-3">
          <div>
            {title && <h2 className="text-base font-semibold leading-tight">{title}</h2>}
            {subtitle && <p className="mt-1 text-xs text-fg-muted">{subtitle}</p>}
          </div>
          {right}
        </header>
      )}
      <div className="px-4 py-3">{children}</div>
      {footer && <footer className="border-t border-edge px-4 py-2 text-xs text-fg-muted">{footer}</footer>}
    </section>
  );
}

export function Field({ label, hint, children, htmlFor }: { label: string; hint?: string; children: ReactNode; htmlFor?: string }) {
  return (
    <div className="mb-3">
      <label htmlFor={htmlFor} className="mb-1 block text-xs font-medium uppercase tracking-wide text-fg-muted">
        {label}
      </label>
      {children}
      {hint && <p className="mt-1 text-xs text-fg-muted">{hint}</p>}
    </div>
  );
}

export function Toggle({ checked, onChange, label, hint, disabled }: { checked: boolean; onChange: (v: boolean) => void; label: ReactNode; hint?: string; disabled?: boolean }) {
  const id = useId();
  return (
    <div className="mb-2 flex items-start gap-2">
      <input
        id={id}
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-[3px] h-4 w-4 flex-none accent-[color:var(--accent)]"
      />
      <label htmlFor={id} className={`text-sm ${disabled ? "text-fg-muted" : ""}`}>
        {label}
        {hint && <span className="block text-xs text-fg-muted">{hint}</span>}
      </label>
    </div>
  );
}

export function Select({ value, onChange, options, id, disabled }: { value: string; onChange: (v: string) => void; options: { value: string; label: string; disabled?: boolean }[]; id?: string; disabled?: boolean }) {
  return (
    <select
      id={id}
      value={value}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value)}
      className="w-full rounded-m border border-edge bg-surface-2 px-2 py-[7px] text-sm"
    >
      {options.map((o) => (
        <option key={o.value} value={o.value} disabled={o.disabled}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

export function NumberInput({ value, onChange, id, min, max, step = 1, placeholder }: { value: number | ""; onChange: (v: number | "") => void; id?: string; min?: number; max?: number; step?: number; placeholder?: string }) {
  return (
    <input
      id={id}
      type="number"
      value={value}
      min={min}
      max={max}
      step={step}
      placeholder={placeholder}
      onChange={(e) => onChange(e.target.value === "" ? "" : Number(e.target.value))}
      className="num w-full rounded-m border border-edge bg-surface-2 px-2 py-[7px] text-sm"
    />
  );
}

/** A term with its definition from this repo's docs. Focus or hover opens it; it is keyboard reachable. */
export function Term({ children, term }: { children?: ReactNode; term: string }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const definition = glossaryFor(term);
  if (!definition) return <>{children ?? term}</>;
  return (
    <span className="relative inline-block">
      <button
        type="button"
        aria-describedby={open ? id : undefined}
        aria-expanded={open}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={() => setOpen((v) => !v)}
        className="cursor-help border-b border-dotted border-fg-muted text-left"
      >
        {children ?? term}
      </button>
      {open && (
        <span
          role="tooltip"
          id={id}
          className="absolute left-0 top-full z-40 mt-1 block w-80 rounded-m border border-edge bg-surface p-3 text-xs leading-relaxed text-fg"
          style={{ boxShadow: "var(--shadow)" }}
        >
          {definition}
        </span>
      )}
    </span>
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="num rounded-s border border-edge bg-surface-2 px-1.5 py-0.5 text-xs text-fg-muted">{children}</kbd>;
}

export function Skeleton({ h = 16, w = "100%" }: { h?: number; w?: number | string }) {
  return <div className="skeleton" style={{ height: h, width: w }} aria-hidden="true" />;
}

export function NotRun({ what }: { what?: string }) {
  return (
    <span className="text-fg-muted">
      not run{what ? <span className="sr-only"> for {what}</span> : null}
    </span>
  );
}
