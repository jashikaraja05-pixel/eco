ALTER TABLE public.scans
  ADD COLUMN IF NOT EXISTS verified_category text CHECK (verified_category IN ('organic','recyclable','hazardous','ewaste','residual','unsure')),
  ADD COLUMN IF NOT EXISTS feedback text CHECK (char_length(feedback) <= 500),
  ADD COLUMN IF NOT EXISTS verified_at timestamptz;
DROP POLICY IF EXISTS "own update" ON public.scans;
CREATE POLICY "own update" ON public.scans FOR UPDATE TO authenticated USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);
-- Only the pilot fields may change; the AI result stays as recorded.
REVOKE UPDATE ON public.scans FROM authenticated;
GRANT UPDATE (verified_category, feedback, verified_at) ON public.scans TO authenticated;