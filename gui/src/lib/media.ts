import { useEffect, useState } from "react";

/** Subscribe to a media query. Used to render one copy of a panel rather than two, because two copies of the
 *  same form mean two elements with the same id, which is invalid and breaks every label association. */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => (typeof window === "undefined" ? false : window.matchMedia(query).matches));
  useEffect(() => {
    const mq = window.matchMedia(query);
    const onChange = () => setMatches(mq.matches);
    onChange();
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, [query]);
  return matches;
}
