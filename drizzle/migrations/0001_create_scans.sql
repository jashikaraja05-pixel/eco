CREATE TABLE IF NOT EXISTS public.scans (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  item text NOT NULL,
  category text NOT NULL CHECK (category IN ('organic','recyclable','hazardous','ewaste','residual','uncertain')),
  bin text NOT NULL,
  confidence integer CHECK (confidence BETWEEN 0 AND 100),
  reasoning text NOT NULL,
  steps jsonb NOT NULL DEFAULT '[]'::jsonb,
  image text NOT NULL,
  source text NOT NULL DEFAULT 'gateway',
  created_at timestamptz NOT NULL DEFAULT now()
);
GRANT SELECT, INSERT, UPDATE, DELETE ON public.scans TO authenticated;
GRANT ALL ON public.scans TO service_role;
ALTER TABLE public.scans ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "own select" ON public.scans;
DROP POLICY IF EXISTS "own insert" ON public.scans;
DROP POLICY IF EXISTS "own delete" ON public.scans;
CREATE POLICY "own select" ON public.scans FOR SELECT TO authenticated USING (auth.uid() = user_id);
CREATE POLICY "own insert" ON public.scans FOR INSERT TO authenticated WITH CHECK (auth.uid() = user_id);
CREATE POLICY "own delete" ON public.scans FOR DELETE TO authenticated USING (auth.uid() = user_id);
CREATE INDEX IF NOT EXISTS scans_user_created ON public.scans (user_id, created_at DESC);