import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState, type ReactNode } from "react";
import {
  Leaf, LogOut, ScanLine, LayoutDashboard, History, BookOpen, Upload, Camera, RotateCcw,
  Loader2, Sparkles, CheckCircle2, Recycle, AlertTriangle, Trash2, Cpu, UserX, HelpCircle, Gauge, Search,
  Eye, EyeOff, Mail, Lock, X, ImagePlus, ArrowRight, Server, Settings, Clock, ShieldCheck,
} from "lucide-react";
import { toast } from "sonner";
import { useServerFn } from "@tanstack/react-start";
import type { User } from "@supabase/supabase-js";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription,
  AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger,
} from "@/components/ui/dialog";
import { supabase } from "@/integrations/supabase/client";
import { classifyWaste, deleteMyAccount } from "@/lib/classify.functions";
import { reuseTips, safetyWarnings, type ScanResult, type StoredCategory } from "@/lib/scan-schema";
import { BINS, rowToScan, stats, type Scan, type ScanRow, type VerifiedCategory } from "@/lib/eco-store";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "EcoLens AI — Scan. Sort. Save." },
      { name: "description", content: "Scan any waste item with AI and learn exactly which bin it belongs in." },
      { property: "og:title", content: "EcoLens AI — Scan. Sort. Save." },
      { property: "og:description", content: "Scan any waste item with AI and learn exactly which bin it belongs in." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: App,
});

type Tab = "scan" | "dashboard" | "history" | "learn";
type Confirm = { title: string; body: string; action: string; run: () => void | Promise<void> };

const NAV: { id: Tab; label: string; icon: typeof Leaf }[] = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "scan", label: "Scan Waste", icon: ScanLine },
  { id: "history", label: "History", icon: History },
  { id: "learn", label: "Learn Hub", icon: BookOpen },
];

const CAT_ICON: Record<StoredCategory, typeof Leaf> = {
  organic: Leaf, recyclable: Recycle, hazardous: AlertTriangle, ewaste: Cpu, residual: Trash2, uncertain: HelpCircle,
};

async function loadHistory(): Promise<Scan[]> {
  const { data, error } = await supabase.from("scans").select("*").order("created_at", { ascending: false }).limit(500);
  if (error) throw new Error("Could not load your history.");
  return (data as ScanRow[]).flatMap((r) => { const s = rowToScan(r); return s ? [s] : []; });
}

const fmtDate = (t: number) => new Date(t).toLocaleString(undefined, { day: "numeric", month: "short", hour: "numeric", minute: "2-digit" });

function App() {
  const [ready, setReady] = useState(false);
  const [user, setUser] = useState<User | null>(null);
  const [history, setHistory] = useState<Scan[]>([]);
  const [tab, setTab] = useState<Tab>("dashboard");
  const [confirm, setConfirm] = useState<Confirm | null>(null);
  const [viewing, setViewing] = useState<Scan | null>(null);
  const [confirmBusy, setConfirmBusy] = useState(false);
  const deleteAccountFn = useServerFn(deleteMyAccount);

  useEffect(() => {
    const { data: sub } = supabase.auth.onAuthStateChange((event, session) => {
      if (event !== "SIGNED_IN" && event !== "SIGNED_OUT" && event !== "INITIAL_SESSION" && event !== "USER_UPDATED") return;
      setUser(session?.user ?? null);
      setReady(true);
      if (session?.user) loadHistory().then(setHistory).catch((e) => toast.error((e as Error).message));
      else setHistory([]);
    });
    return () => sub.subscription.unsubscribe();
  }, []);

  const go = (t: Tab) => { setTab(t); window.scrollTo({ top: 0 }); };
  const logout = async () => { await supabase.auth.signOut(); setTab("dashboard"); };
  const removeScan = (id: string) => setConfirm({
    title: "Delete this scan?", body: "It will be permanently removed from your history. This cannot be undone.", action: "Delete scan",
    run: async () => {
      const { error } = await supabase.from("scans").delete().eq("id", id);
      if (error) { toast.error("Could not delete scan."); return; }
      setHistory((h) => h.filter((s) => s.id !== id)); toast.success("Scan deleted.");
    },
  });
  const clearAll = () => user && setConfirm({
    title: "Clear your entire history?", body: `All ${history.length} saved scans will be permanently deleted. This cannot be undone.`, action: "Clear history",
    run: async () => {
      const { error } = await supabase.from("scans").delete().eq("user_id", user.id);
      if (error) { toast.error("Could not clear history."); return; }
      setHistory([]); toast.success("History cleared.");
    },
  });
  const removeAccount = () => user && setConfirm({
    title: "Delete your account?", body: "Your account and every saved scan will be permanently deleted. This cannot be undone.", action: "Delete account",
    run: async () => {
      try {
        await deleteAccountFn();
        await supabase.auth.signOut();
        toast.success("Your account has been deleted.");
      } catch (err) { toast.error((err as Error).message); }
    },
  });

  const runConfirm = async () => {
    if (!confirm) return;
    setConfirmBusy(true);
    try { await confirm.run(); } finally { setConfirmBusy(false); setConfirm(null); }
  };

  if (!ready) return <div className="grid min-h-dvh place-items-center"><Loader2 className="size-6 animate-spin text-primary" aria-label="Loading" /></div>;
  if (!user) return <AuthScreen />;

  return (
    <div className="min-h-dvh">
      <TopNav tab={tab} setTab={go} user={user} onLogout={logout} onDeleteAccount={removeAccount} />
      <main className="mx-auto w-full max-w-6xl px-4 pb-12 pt-5 sm:px-6 sm:pt-8 lg:px-10">
        <div key={tab} className="fade-up">
          {tab === "scan" && <Scanner onSaved={(s) => setHistory((h) => [s, ...h])} />}
          {tab === "dashboard" && <Dashboard history={history} />}
          {tab === "history" && <HistoryView history={history} onDelete={removeScan} onClearAll={clearAll} onGoScan={() => go("scan")} onView={setViewing} />}
          {tab === "learn" && <Learn />}
        </div>
      </main>
      <ScanDetail scan={viewing} onClose={() => setViewing(null)} onVerified={(s) => { setViewing(s); setHistory((h) => h.map((x) => (x.id === s.id ? s : x))); }} />
      <AlertDialog open={!!confirm} onOpenChange={(o) => { if (!o && !confirmBusy) setConfirm(null); }}>
        <AlertDialogContent className="w-[calc(100%-2rem)] max-w-md rounded-2xl">
          <AlertDialogHeader>
            <span className="mx-auto grid size-12 place-items-center rounded-full bg-destructive/10 text-destructive sm:mx-0"><Trash2 className="size-5" /></span>
            <AlertDialogTitle>{confirm?.title}</AlertDialogTitle>
            <AlertDialogDescription>{confirm?.body}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter className="gap-2">
            <AlertDialogCancel disabled={confirmBusy} className="h-11 rounded-xl">Cancel</AlertDialogCancel>
            <AlertDialogAction disabled={confirmBusy} onClick={(e) => { e.preventDefault(); void runConfirm(); }}
              className="h-11 rounded-xl bg-destructive text-destructive-foreground hover:bg-destructive/90">
              {confirmBusy && <Loader2 className="size-4 animate-spin" />}{confirm?.action}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

/* ---------------- Navigation ---------------- */
function Brand({ tagline = false }: { tagline?: boolean }) {
  return (
    <div className="flex min-w-0 items-center gap-2.5">
      <span className="bg-ocean glow grid size-9 shrink-0 place-items-center rounded-xl text-primary-foreground">
        <Leaf className="size-5" />
      </span>
      <div className="min-w-0 leading-tight">
        <span className="font-display block truncate text-lg font-bold">EcoLens <span className="text-primary">AI</span></span>
        {tagline && <span className="block text-xs text-muted-foreground">Scan. Sort. Save.</span>}
      </div>
    </div>
  );
}

function AccountSettings({ user, onLogout, onDeleteAccount }: { user: User; onLogout: () => void; onDeleteAccount: () => void }) {
  const [open, setOpen] = useState(false);
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="ghost" size="icon" className="size-10 shrink-0 rounded-full" aria-label="Account settings"><Settings className="size-5" /></Button>
      </DialogTrigger>
      <DialogContent className="w-[calc(100%-2rem)] max-w-sm rounded-3xl">
        <DialogHeader>
          <DialogTitle>Account Settings</DialogTitle>
          <DialogDescription>Signed in as</DialogDescription>
        </DialogHeader>
        <p className="flex items-center gap-2 truncate rounded-xl bg-muted px-3 py-3 text-sm"><Mail className="size-4 shrink-0 text-primary" /><span className="truncate">{user.email}</span></p>
        <div className="grid gap-2">
          <Button variant="outline" className="h-11 rounded-xl" onClick={() => { setOpen(false); onLogout(); }}><LogOut className="size-4" /> Log Out</Button>
          <Button variant="outline" className="h-11 rounded-xl text-destructive hover:text-destructive" onClick={() => { setOpen(false); onDeleteAccount(); }}><UserX className="size-4" /> Delete Account</Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function TopNav({ tab, setTab, user, onLogout, onDeleteAccount }: { tab: Tab; setTab: (t: Tab) => void; user: User; onLogout: () => void; onDeleteAccount: () => void }) {
  return (
    <header className="pt-safe sticky top-0 z-40 border-b border-border bg-background/85 backdrop-blur">
      <div className="mx-auto flex h-16 max-w-6xl items-center gap-4 px-4 sm:px-6 lg:px-10">
        <Brand />
        <nav aria-label="Main" className="ml-auto hidden items-center gap-1 rounded-full bg-muted p-1 md:flex">
          {NAV.map(({ id, label, icon: Icon }) => (
            <button key={id} onClick={() => setTab(id)} aria-current={tab === id ? "page" : undefined}
              className={`flex h-9 items-center gap-2 rounded-full px-4 text-sm font-medium transition-all ${tab === id ? "bg-card text-primary shadow-sm" : "text-muted-foreground hover:text-foreground"}`}>
              <Icon className="size-4" /> {label}
            </button>
          ))}
        </nav>
        <div className="ml-auto md:ml-0"><AccountSettings user={user} onLogout={onLogout} onDeleteAccount={onDeleteAccount} /></div>
      </div>
      <nav aria-label="Main" className="no-scrollbar grid grid-cols-4 gap-1 px-2 pb-2 md:hidden">
        {NAV.map(({ id, label, icon: Icon }) => (
          <button key={id} onClick={() => setTab(id)} aria-current={tab === id ? "page" : undefined}
            className={`flex min-h-12 min-w-0 flex-col items-center justify-center gap-0.5 rounded-xl text-[11px] font-medium transition-colors ${tab === id ? "bg-secondary text-primary" : "text-muted-foreground"}`}>
            <Icon className="size-5" /><span className="max-w-full truncate">{label}</span>
          </button>
        ))}
      </nav>
    </header>
  );
}

function PageHeader({ title, sub, action }: { title: string; sub: ReactNode; action?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
      <div className="min-w-0">
        <h1 className="text-2xl font-bold sm:text-3xl">{title}</h1>
        <p className="mt-1 text-sm text-muted-foreground sm:text-base">{sub}</p>
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </div>
  );
}

function EmptyState({ icon: Icon, title, body, action }: { icon: typeof Leaf; title: string; body: string; action?: ReactNode }) {
  return (
    <div className="glass bg-soft flex flex-col items-center rounded-3xl px-6 py-12 text-center sm:py-16">
      <span className="relative grid size-20 place-items-center rounded-full bg-secondary">
        <span className="absolute inset-0 animate-ping rounded-full bg-primary/10" />
        <Icon className="size-9 text-primary" />
      </span>
      <p className="mt-5 text-lg font-semibold">{title}</p>
      <p className="mt-1 max-w-sm text-sm text-muted-foreground">{body}</p>
      {action && <div className="mt-6">{action}</div>}
    </div>
  );
}

/* ---------------- Auth ---------------- */
function AuthScreen() {
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [show, setShow] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null); setInfo(null);
    const em = email.trim().toLowerCase();
    if (!/^\S+@\S+\.\S+$/.test(em)) { setError("Please enter a valid email address."); return; }
    if (password.length < 8) { setError("Password must be at least 8 characters."); return; }
    setBusy(true);
    try {
      if (mode === "signup") {
        const { data, error } = await supabase.auth.signUp({ email: em, password, options: { emailRedirectTo: window.location.origin } });
        if (error) throw error;
        if (!data.session) setInfo("Check your inbox to confirm your email, then log in.");
      } else {
        const { error } = await supabase.auth.signInWithPassword({ email: em, password });
        if (error) throw new Error(error.message.includes("confirm") ? "Please confirm your email first." : "Account not registered or invalid password. Please sign up first.");
      }
    } catch (err) {
      setError((err as Error).message);
    } finally { setBusy(false); }
  };

  return (
    <div className="bg-auth flex min-h-dvh items-center justify-center">
      <div className="pt-safe flex w-full items-center justify-center px-4 py-10 sm:px-6">
        <div className="fade-up w-full max-w-md rounded-3xl border border-border bg-card p-6 shadow-xl sm:p-8">
          <div className="mb-6 flex justify-center"><Brand tagline /></div>
          <div className="grid grid-cols-2 rounded-xl bg-muted p-1" role="tablist" aria-label="Authentication mode">
            {(["login", "signup"] as const).map((m) => (
              <button key={m} type="button" role="tab" aria-selected={mode === m} onClick={() => { setMode(m); setError(null); setInfo(null); }}
                className={`h-10 rounded-lg text-sm font-semibold transition-all ${mode === m ? "bg-card text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}>
                {m === "login" ? "Log In" : "Sign Up"}
              </button>
            ))}
          </div>
          <h2 className="mt-6 text-2xl font-bold">{mode === "login" ? "Welcome back" : "Create your account"}</h2>
          <p className="mt-1 text-sm text-muted-foreground">{mode === "login" ? "Log in to continue tracking your impact." : "Start sorting smarter in under a minute."}</p>
          <form onSubmit={submit} className="mt-6 space-y-4" noValidate>
            <div className="space-y-1.5">
              <Label htmlFor="email">Email</Label>
              <div className="relative">
                <Mail className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" aria-hidden="true" />
                <Input id="email" className="h-12 rounded-xl pl-10 text-base sm:text-sm" autoComplete="email" type="email" inputMode="email" required placeholder="you@example.com" value={email} onChange={(e) => setEmail(e.target.value)} aria-invalid={!!error} />
              </div>
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="password">Password</Label>
              <div className="relative">
                <Lock className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" aria-hidden="true" />
                <Input id="password" className="h-12 rounded-xl px-10 text-base sm:text-sm" autoComplete={mode === "signup" ? "new-password" : "current-password"} type={show ? "text" : "password"} required placeholder="At least 8 characters" value={password} onChange={(e) => setPassword(e.target.value)} aria-invalid={!!error} aria-describedby="pw-hint" />
                <button type="button" onClick={() => setShow((s) => !s)} aria-label={show ? "Hide password" : "Show password"}
                  className="absolute right-1 top-1/2 grid size-10 -translate-y-1/2 place-items-center rounded-lg text-muted-foreground hover:text-foreground">
                  {show ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
                </button>
              </div>
              {mode === "signup" && <p id="pw-hint" className="text-xs text-muted-foreground">Use 8 or more characters.</p>}
            </div>
            {info && <p role="status" className="flex gap-2 rounded-xl border border-primary/30 bg-secondary p-3 text-sm text-secondary-foreground"><Mail className="mt-0.5 size-4 shrink-0" />{info}</p>}
            {error && <p role="alert" className="flex gap-2 rounded-xl border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive"><AlertTriangle className="mt-0.5 size-4 shrink-0" />{error}</p>}
            <Button type="submit" disabled={busy} className="glow h-12 w-full rounded-xl text-base">
              {busy ? <><Loader2 className="size-4 animate-spin" /> {mode === "login" ? "Logging in…" : "Creating account…"}</> : <>{mode === "login" ? "Log In" : "Create Account"} <ArrowRight className="size-4" /></>}
            </Button>
          </form>
          <p className="mt-6 flex items-center justify-center gap-1.5 text-xs text-muted-foreground"><ShieldCheck className="size-3.5" /> Your scans are private to your account.</p>
        </div>
      </div>
    </div>
  );
}

/* ---------------- Scanner ---------------- */
function shrink(src: string, max = 768): Promise<string> {
  return new Promise((res, rej) => {
    const img = new Image();
    img.onload = () => {
      const s = Math.min(1, max / Math.max(img.width, img.height));
      const c = document.createElement("canvas");
      c.width = Math.round(img.width * s);
      c.height = Math.round(img.height * s);
      const context = c.getContext("2d");
      if (!context) { rej(new Error("Image processing is unavailable.")); return; }
      context.drawImage(img, 0, 0, c.width, c.height);
      res(c.toDataURL("image/jpeg", 0.82));
    };
    img.onerror = () => rej(new Error("Could not read this image."));
    img.src = src;
  });
}

const STAGES = ["Uploading image", "Detecting the item", "Matching segregation rules"];

function Scanner({ onSaved }: { onSaved: (s: Scan) => void }) {
  const [mode, setMode] = useState<"upload" | "camera">("upload");
  const [image, setImage] = useState<string | null>(null);
  const [result, setResult] = useState<ScanResult | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const classify = useServerFn(classifyWaste);
  const [loading, setLoading] = useState(false);
  const [stage, setStage] = useState(0);
  const [drag, setDrag] = useState(false);
  const [camOn, setCamOn] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const operationRef = useRef(0);
  const cameraRequestRef = useRef(0);
  const analyzingRef = useRef(false);

  useEffect(() => {
    if (!loading) { setStage(0); return; }
    const t = setInterval(() => setStage((s) => Math.min(s + 1, STAGES.length - 1)), 1400);
    return () => clearInterval(t);
  }, [loading]);

  const stopCam = () => {
    cameraRequestRef.current += 1;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCamOn(false);
  };
  useEffect(() => () => { operationRef.current += 1; stopCam(); }, []);

  const startCam = async () => {
    stopCam();
    const request = cameraRequestRef.current;
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
      if (request !== cameraRequestRef.current) { s.getTracks().forEach((t) => t.stop()); return; }
      streamRef.current = s;
      setCamOn(true);
      requestAnimationFrame(() => {
        if (request !== cameraRequestRef.current) return;
        if (videoRef.current) {
          videoRef.current.srcObject = s;
          void videoRef.current.play().catch(() => { stopCam(); toast.error("Camera preview could not start."); });
        }
      });
    } catch {
      if (request !== cameraRequestRef.current) return;
      toast.error("Camera access was denied or is unavailable.");
    }
  };

  const capture = async () => {
    const v = videoRef.current;
    if (!v) return;
    const c = document.createElement("canvas");
    c.width = v.videoWidth; c.height = v.videoHeight;
    const context = c.getContext("2d");
    if (!context || !v.videoWidth || !v.videoHeight) { toast.error("Camera is not ready yet."); return; }
    context.drawImage(v, 0, 0);
    setImage(await shrink(c.toDataURL("image/jpeg", 0.9)));
    stopCam();
  };

  const onFile = (f?: File) => {
    if (!f || analyzingRef.current) return;
    if (!f.type.startsWith("image/")) { toast.error("Please choose an image file."); return; }
    const operation = ++operationRef.current;
    setImage(null); setResult(null); setFailure(null);
    const r = new FileReader();
    r.onload = async () => {
      try {
        if (typeof r.result !== "string") throw new Error("Could not read this image.");
        const processed = await shrink(r.result);
        if (operation === operationRef.current) setImage(processed);
      } catch (e) { if (operation === operationRef.current) toast.error((e as Error).message); }
    };
    r.onerror = () => { if (operation === operationRef.current) toast.error("Could not read this image."); };
    r.readAsDataURL(f);
  };

  const analyze = async () => {
    if (!image || analyzingRef.current) return;
    const operation = ++operationRef.current;
    analyzingRef.current = true;
    setLoading(true); setFailure(null);
    try {
      const res = await classify({ data: { image } });
      if (operation !== operationRef.current) return;
      onSaved(res);
      setResult(res);
      if (res.category === "uncertain") toast.warning("Low confidence — please rescan."); else toast.success("Classified and saved.");
    } catch (e) {
      if (operation === operationRef.current) setFailure((e as Error).message || "Analysis failed. Please try again.");
    } finally {
      analyzingRef.current = false;
      if (operation === operationRef.current) setLoading(false);
    }
  };

  const reset = () => {
    if (analyzingRef.current) return;
    operationRef.current += 1;
    stopCam();
    setImage(null);
    setResult(null); setFailure(null);
    if (fileRef.current) fileRef.current.value = "";
  };

  const switchMode = (m: "upload" | "camera") => { reset(); setMode(m); };

  return (
    <section>
      <PageHeader title="Scan Waste" sub="Upload or capture a photo — we'll tell you which bin it goes in."
        action={result && <Button onClick={reset} size="lg" className="fade-up h-11 w-full rounded-xl sm:w-auto"><RotateCcw className="size-4" /> New Scan</Button>} />

      <div className="grid gap-5 lg:grid-cols-2 lg:gap-6">
        <div className="glass min-w-0 rounded-3xl p-4 sm:p-6">
          <div className="mb-4 grid grid-cols-2 gap-1 rounded-xl bg-muted p-1" role="tablist" aria-label="Image source">
            {(["upload", "camera"] as const).map((m) => (
              <button type="button" role="tab" aria-selected={mode === m} disabled={loading} key={m} onClick={() => switchMode(m)}
                className={`flex h-11 items-center justify-center gap-2 rounded-lg text-sm font-medium transition-all disabled:opacity-50 ${mode === m ? "bg-card text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"}`}>
                {m === "upload" ? <Upload className="size-4" /> : <Camera className="size-4" />}
                {m === "upload" ? "Upload Photo" : "Use Camera"}
              </button>
            ))}
          </div>

          <div className={`relative aspect-[4/3] overflow-hidden rounded-2xl border-2 transition-colors ${image || camOn ? "border-transparent bg-foreground/5" : drag ? "border-dashed border-primary bg-secondary" : "border-dashed border-input bg-muted/50"}`}>
            {image ? (
              <img src={image} alt="Waste item preview" className="size-full object-contain" />
            ) : mode === "camera" && camOn ? (
              <video ref={videoRef} playsInline muted className="size-full object-cover" />
            ) : mode === "upload" ? (
              <button type="button" onClick={() => fileRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
                onDragLeave={() => setDrag(false)}
                onDrop={(e) => { e.preventDefault(); setDrag(false); onFile(e.dataTransfer.files[0]); }}
                className="group flex size-full flex-col items-center justify-center gap-3 p-4 text-center">
                <span className={`grid size-16 place-items-center rounded-2xl transition-all ${drag ? "bg-primary text-primary-foreground" : "bg-card text-primary shadow-sm group-hover:scale-105"}`}><ImagePlus className="size-8" /></span>
                <span className="font-semibold">{drag ? "Drop to upload" : "Tap to choose a photo"}</span>
                <span className="hidden text-sm text-muted-foreground sm:block">or drag and drop an image here</span>
                <span className="text-xs text-muted-foreground">JPG, PNG or WEBP · one item per photo works best</span>
              </button>
            ) : (
              <button type="button" onClick={startCam} className="group flex size-full flex-col items-center justify-center gap-3 p-4 text-center">
                <span className="grid size-16 place-items-center rounded-2xl bg-card text-primary shadow-sm transition-transform group-hover:scale-105"><Camera className="size-8" /></span>
                <span className="font-semibold">Start camera</span>
                <span className="text-xs text-muted-foreground">You'll be asked for camera permission</span>
              </button>
            )}
            {image && !loading && !result && (
              <Button type="button" variant="secondary" size="icon" onClick={reset} aria-label="Remove image" className="absolute right-3 top-3 size-10 rounded-full shadow-md"><X className="size-4" /></Button>
            )}
            {loading && (
              <div className="absolute inset-0 bg-foreground/30 backdrop-blur-[2px]" role="status" aria-live="polite">
                <div className="scan-line absolute inset-x-6 h-0.5 rounded-full bg-primary shadow-[0_0_16px_2px] shadow-primary" />
                <div className="absolute inset-x-3 bottom-3 rounded-2xl bg-card p-4 shadow-lg sm:inset-x-6 sm:bottom-6">
                  <p className="flex items-center gap-2 text-sm font-semibold"><Loader2 className="size-4 animate-spin text-primary" /> {STAGES[stage]}…</p>
                  <div className="mt-3 grid grid-cols-3 gap-1.5">
                    {STAGES.map((s, i) => <span key={s} className={`h-1.5 rounded-full transition-colors duration-500 ${i <= stage ? "bg-primary" : "bg-muted"}`} />)}
                  </div>
                  <p className="mt-2 text-xs text-muted-foreground">This usually takes a few seconds.</p>
                </div>
              </div>
            )}
          </div>
          <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={(e) => onFile(e.target.files?.[0])} />

          <div className="mt-4 flex flex-col gap-2 sm:flex-row sm:gap-3">
            {mode === "camera" && camOn && !image && (
              <>
                <Button className="h-12 flex-1 rounded-xl text-base" onClick={capture}><Camera className="size-5" /> Capture Photo</Button>
                <Button variant="outline" className="h-12 rounded-xl" onClick={stopCam}>Stop Camera</Button>
              </>
            )}
            {image && !result && (
              <Button className="glow h-12 flex-1 rounded-xl text-base" onClick={analyze} disabled={loading}>
                {loading ? <Loader2 className="size-5 animate-spin" /> : <Sparkles className="size-5" />} {loading ? "Analyzing…" : "Analyze Waste"}
              </Button>
            )}
            {image && !result && !loading && (
              <Button variant="outline" className="h-12 rounded-xl" onClick={() => mode === "camera" ? (setImage(null), void startCam()) : fileRef.current?.click()}>
                {mode === "camera" ? <Camera className="size-4" /> : <Upload className="size-4" />} {mode === "camera" ? "Retake" : "Replace"}
              </Button>
            )}
          </div>
        </div>

        <div className="min-w-0">
          {result ? <ResultCard r={result} /> : failure ? (
            <div role="alert" className="fade-up flex h-full min-h-72 flex-col items-center justify-center rounded-3xl border-2 border-destructive/30 bg-destructive/5 p-8 text-center">
              <span className="grid size-16 place-items-center rounded-full bg-destructive/10"><AlertTriangle className="size-8 text-destructive" /></span>
              <p className="mt-4 text-lg font-semibold text-destructive">Analysis failed</p>
              <p className="mt-1 max-w-xs text-sm text-muted-foreground">{failure}</p>
              <p className="mt-2 text-xs text-muted-foreground">No result was saved.</p>
              <Button variant="outline" className="mt-5 h-11 rounded-xl" onClick={analyze} disabled={!image || loading}><RotateCcw className="size-4" /> Try again</Button>
            </div>
          ) : (
            <div className="glass bg-soft flex h-full min-h-72 flex-col items-center justify-center rounded-3xl p-8 text-center">
              <span className="grid size-16 place-items-center rounded-2xl bg-secondary"><ScanLine className="size-8 text-primary" /></span>
              <p className="mt-4 font-semibold">Results appear here</p>
              <p className="mt-1 max-w-xs text-sm text-muted-foreground">You'll see the category, recommended bin, confidence and step-by-step disposal guidance.</p>
              <ol className="mt-6 grid w-full max-w-xs gap-2 text-left text-sm">
                {["Add a clear photo of one item", "Tap Analyze Waste", "Follow the disposal steps"].map((s, i) => (
                  <li key={s} className="flex items-center gap-3 rounded-xl bg-card px-3 py-2.5 shadow-sm"><span className="grid size-6 shrink-0 place-items-center rounded-full bg-primary text-xs font-bold text-primary-foreground">{i + 1}</span>{s}</li>
                ))}
              </ol>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

function CategoryBadge({ c, compact = false }: { c: ScanResult["category"]; compact?: boolean }) {
  const b = BINS[c];
  const Icon = CAT_ICON[c];
  return (
    <span className={`${b.tone} inline-flex max-w-full shrink-0 items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold text-primary-foreground`}>
      <Icon className="size-3.5 shrink-0" aria-hidden="true" />
      <span className="truncate">{b.label}{!compact && ` · ${b.bin}`}</span>
    </span>
  );
}

function ResultCard({ r }: { r: ScanResult }) {
  const unsure = r.category === "uncertain";
  const b = BINS[r.category];
  const Icon = CAT_ICON[r.category];
  return (
    <div className={`fade-up overflow-hidden rounded-3xl border bg-card shadow-sm ${unsure ? "border-warning" : "border-border"}`}>
      <div className={`${unsure ? "bg-warning-soft text-warning-foreground" : `${b.tone} text-primary-foreground`} flex items-center gap-4 p-5 sm:p-6`}>
        <span className="grid size-14 shrink-0 place-items-center rounded-2xl bg-card/20"><Icon className="size-7" /></span>
        <div className="min-w-0">
          <p className="text-xs font-semibold uppercase tracking-wider opacity-85">{unsure ? "Needs a rescan" : "Goes in the"}</p>
          <p className="font-display text-2xl font-bold">{unsure ? "Uncertain" : b.bin}</p>
          <p className="text-sm opacity-90">{b.label}</p>
        </div>
      </div>
      <div className="p-5 sm:p-6">
        <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Detected item</p>
        <h2 className="mt-1 text-xl font-bold sm:text-2xl">{r.item}</h2>
        <p className="mt-1 text-sm font-medium text-primary">{r.bin}</p>
        {unsure && <p role="status" className="mt-3 flex gap-2 rounded-xl border border-warning/50 bg-warning-soft p-3 text-sm text-warning-foreground"><HelpCircle className="mt-0.5 size-4 shrink-0" /> The model isn't confident enough to recommend a bin. Retake the photo with one item in clear light.</p>}
        {safetyWarnings[r.category].length > 0 && (
          <div role={r.category === "hazardous" || r.category === "ewaste" ? "alert" : "note"} className={`mt-3 space-y-1.5 rounded-xl border p-3 text-sm ${r.category === "hazardous" || r.category === "ewaste" ? "border-destructive/40 bg-destructive/10 text-destructive" : "border-border bg-muted text-muted-foreground"}`}>
            {safetyWarnings[r.category].map((w) => <p key={w} className="flex gap-2"><AlertTriangle className="mt-0.5 size-4 shrink-0" />{w}</p>)}
          </div>
        )}
        <div className="mt-5">
          <div className="flex justify-between text-sm"><span className="text-muted-foreground">Model confidence</span><span className="font-semibold">{r.confidence}%</span></div>
          <div className="mt-2 h-2 overflow-hidden rounded-full bg-muted" role="progressbar" aria-valuenow={r.confidence} aria-valuemin={0} aria-valuemax={100} aria-label="Model confidence">
            <div className={`${unsure ? "bg-warning" : "bg-ocean"} h-full rounded-full transition-all duration-700`} style={{ width: `${r.confidence}%` }} />
          </div>
        </div>
        <div className="mt-5 rounded-2xl bg-secondary p-4 text-sm"><p className="mb-1 font-semibold">Why this bin?</p><p className="text-secondary-foreground">{r.reasoning}</p></div>
        <p className="mt-5 font-semibold">Disposal guidance</p>
        <ol className="mt-3 space-y-2.5">
          {r.steps.map((s, i) => (
            <li key={i} className="flex gap-3 text-sm"><span className="grid size-6 shrink-0 place-items-center rounded-full bg-secondary text-xs font-bold text-secondary-foreground">{i + 1}</span><span className="pt-0.5">{s}</span></li>
          ))}
        </ol>
        <div className="mt-5 rounded-2xl border border-border p-4 text-sm"><p className="mb-1 flex items-center gap-1.5 font-semibold"><Recycle className="size-4 text-primary" /> Reuse &amp; recycling</p><p className="text-muted-foreground">{reuseTips[r.category]}</p></div>
        <p className="mt-5 flex items-center gap-1.5 border-t border-border pt-4 text-xs text-muted-foreground"><Server className="size-3.5" /> Engine: {r.source === "python" ? "Python FastAPI service" : "Built-in vision AI"} · <CheckCircle2 className="size-3.5 text-primary" /> Saved to history</p>
      </div>
    </div>
  );
}

/* ---------------- Dashboard ---------------- */
function Dashboard({ history }: { history: Scan[] }) {
  const s = stats(history);
  const cards = [
    { label: "Total Scans", value: s.total, icon: ScanLine, tone: "bg-ocean" },
    { label: "Organic", value: s.organic, icon: Leaf, tone: "bg-bin-green" },
    { label: "Recyclable", value: s.recyclable, icon: Recycle, tone: "bg-bin-blue" },
    { label: "Hazardous", value: s.hazardous, icon: AlertTriangle, tone: "bg-bin-red" },
    { label: "E-Waste", value: s.ewaste, icon: Cpu, tone: "bg-bin-red" },
    { label: "Residual", value: s.residual, icon: Trash2, tone: "bg-bin-black" },
    { label: "Uncertain", value: s.uncertain, icon: HelpCircle, tone: "bg-muted-foreground" },
  ];
  const bars = cards.slice(1);
  const days = new Set(history.map((h) => new Date(h.createdAt).toDateString())).size;
  const max = Math.max(1, ...bars.map((b) => b.value));
  return (
    <section>
      <PageHeader title="Dashboard" sub="Your sorting at a glance." />
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 sm:gap-4 lg:grid-cols-7">
        {cards.map(({ label, value, icon: Icon, tone }, i) => (
          <div key={label} className={`glass lift min-w-0 rounded-3xl p-4 sm:p-5 ${i === 0 ? "col-span-2 sm:col-span-1" : ""}`}>
            <span className={`${tone} grid size-10 place-items-center rounded-xl text-primary-foreground`}><Icon className="size-5" /></span>
            <p className="mt-3 text-2xl font-bold sm:text-3xl">{value}</p>
            <p className="truncate text-sm text-muted-foreground">{label}</p>
          </div>
        ))}
      </div>
      <div className="glass mt-6 rounded-3xl p-5 sm:p-6">
        <p className="font-semibold">Sorting by category</p>
        <p className="text-sm text-muted-foreground">{s.total === 0 ? "Scan an item to start your chart." : `Based on your ${s.total} scan${s.total === 1 ? "" : "s"}`}</p>
        <div className="mt-6 flex h-56 items-end justify-around gap-3 border-b border-border sm:gap-6" role="img"
          aria-label={bars.map((b) => `${b.label} ${b.value}`).join(", ")}>
          {bars.map((b) => (
            <div key={b.label} className="flex h-full min-w-0 flex-1 flex-col items-center justify-end gap-2">
              <span className="text-sm font-semibold tabular-nums">{b.value}</span>
              <div className={`${b.tone} w-full max-w-16 rounded-t-xl transition-all duration-700`} style={{ height: `${Math.max(2, (b.value / max) * 85)}%` }} />
            </div>
          ))}
        </div>
        <div className="mt-2 flex justify-around gap-3 sm:gap-6">
          {bars.map((b) => <span key={b.label} className="min-w-0 flex-1 truncate text-center text-xs text-muted-foreground sm:text-sm">{b.label}</span>)}
        </div>
      </div>
      <div className="mt-6 grid gap-3 sm:grid-cols-3 sm:gap-4">
        <div className="glass rounded-3xl p-5"><p className="text-sm text-muted-foreground">Active days</p><p className="mt-1 text-2xl font-bold">{days}</p><p className="text-xs text-muted-foreground">Days you scanned at least once</p></div>
        <div className="glass rounded-3xl p-5"><p className="text-sm text-muted-foreground">Avg. confidence</p><p className="mt-1 text-2xl font-bold">{s.total - s.uncertain ? `${s.avgConfidence}%` : "—"}</p><p className="text-xs text-muted-foreground">Confident results only</p></div>
        <div className="glass rounded-3xl p-5"><p className="flex items-center gap-1.5 text-sm text-muted-foreground"><ShieldCheck className="size-4" /> Pilot accuracy</p><p className="mt-1 text-2xl font-bold">{s.pilotAccuracy === null ? "—" : `${s.pilotAccuracy}%`}</p><p className="text-xs text-muted-foreground">{s.verified ? `${s.verifiedCorrect} of ${s.verified} scans you checked were right` : "Open a scan in History and mark the real category"}</p></div>
      </div>
    </section>
  );
}

const VERIFY_OPTS: { id: VerifiedCategory; label: string }[] = [
  { id: "organic", label: "Organic" }, { id: "recyclable", label: "Recyclable" }, { id: "hazardous", label: "Hazardous" },
  { id: "ewaste", label: "E-Waste" }, { id: "residual", label: "Residual" }, { id: "unsure", label: "Not sure" },
];

function PilotCheck({ scan, onVerified }: { scan: Scan; onVerified: (s: Scan) => void }) {
  const [choice, setChoice] = useState<VerifiedCategory | null>(scan.verifiedCategory);
  const [note, setNote] = useState(scan.feedback ?? "");
  const [busy, setBusy] = useState(false);
  const save = async () => {
    if (!choice) return;
    setBusy(true);
    const { error } = await supabase.from("scans").update({ verified_category: choice, feedback: note.trim() || null, verified_at: new Date().toISOString() }).eq("id", scan.id);
    setBusy(false);
    if (error) { toast.error("Could not save your check."); return; }
    onVerified({ ...scan, verifiedCategory: choice, feedback: note.trim() || null });
    toast.success("Pilot check saved.");
  };
  return (
    <div className="mx-6 mb-4 rounded-2xl border border-border p-4">
      <p className="font-semibold">Pilot check: what is it really?</p>
      <p className="text-xs text-muted-foreground">Your answer is used only to measure real-world accuracy on your dashboard.</p>
      <div className="mt-3 flex flex-wrap gap-2" role="radiogroup" aria-label="Actual category">
        {VERIFY_OPTS.map((o) => (
          <Button key={o.id} type="button" size="sm" variant={choice === o.id ? "default" : "outline"} role="radio" aria-checked={choice === o.id} onClick={() => setChoice(o.id)}>{o.label}</Button>
        ))}
      </div>
      <Label htmlFor="pilot-note" className="mt-3 block text-xs">Feedback (optional)</Label>
      <Input id="pilot-note" maxLength={500} value={note} onChange={(e) => setNote(e.target.value)} placeholder="e.g. it was a chip packet, not paper" className="mt-1" />
      <Button type="button" size="sm" className="mt-3" disabled={!choice || busy} onClick={save}>{busy ? <Loader2 className="size-4 animate-spin" /> : "Save check"}</Button>
    </div>
  );
}

function ScanDetail({ scan, onClose, onVerified }: { scan: Scan | null; onClose: () => void; onVerified: (s: Scan) => void }) {
  return (
    <Dialog open={!!scan} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-h-[90dvh] w-[calc(100%-2rem)] max-w-lg overflow-y-auto rounded-3xl p-0">
        <DialogHeader className="sr-only"><DialogTitle>{scan?.item ?? "Scan"}</DialogTitle><DialogDescription>Scan details</DialogDescription></DialogHeader>
        {scan && (
          <>
            <img src={scan.image} alt={scan.item} className="aspect-video w-full rounded-t-3xl bg-muted object-cover" />
            <div className="px-1 pb-1"><ResultCard r={scan} /></div>
            <PilotCheck key={scan.id} scan={scan} onVerified={onVerified} />
            <p className="flex items-center gap-1 px-6 pb-5 text-xs text-muted-foreground"><Clock className="size-3" />{fmtDate(scan.createdAt)}</p>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}

/* ---------------- History ---------------- */
function HistoryView({ history, onDelete, onClearAll, onGoScan, onView }: { history: Scan[]; onDelete: (id: string) => void; onClearAll: () => void; onGoScan: () => void; onView: (s: Scan) => void }) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<StoredCategory | "all">("all");
  const q = query.trim().toLowerCase();
  const shown = history.filter((h) => (filter === "all" || h.category === filter) && (!q || h.item.toLowerCase().includes(q)));
  const filters: { id: StoredCategory | "all"; label: string; n: number }[] = [
    { id: "all", label: "All", n: history.length },
    ...(Object.keys(BINS) as StoredCategory[]).map((c) => ({ id: c, label: BINS[c].label, n: history.filter((h) => h.category === c).length })),
  ];
  return (
    <section>
      <PageHeader title="Scan History"
        sub={shown.length === history.length
          ? `${history.length} scan${history.length === 1 ? "" : "s"} saved to your account.`
          : `${shown.length} of ${history.length} scan${history.length === 1 ? "" : "s"} shown.`}
        action={history.length > 0 && (
          <Button variant="outline" className="h-11 w-full rounded-xl text-destructive hover:text-destructive sm:w-auto" onClick={onClearAll}>
            <Trash2 className="size-4" /> Clear All History
          </Button>
        )} />
      {history.length > 0 && (
        <div className="glass space-y-3 rounded-2xl p-3 sm:p-4">
          <div className="relative">
            <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" aria-hidden="true" />
            <Input className="h-11 rounded-xl pl-10 text-base sm:text-sm" aria-label="Search scans" type="search" placeholder="Search by item name…" value={query} onChange={(e) => setQuery(e.target.value)} />
          </div>
          <div className="no-scrollbar -mx-3 flex gap-2 overflow-x-auto px-3 sm:mx-0 sm:flex-wrap sm:px-0" role="group" aria-label="Filter by category">
            {filters.map(({ id, label, n }) => (
              <button type="button" key={id} aria-pressed={filter === id} onClick={() => setFilter(id)}
                className={`flex h-9 shrink-0 items-center gap-1.5 rounded-full border px-3.5 text-sm font-medium transition-colors ${filter === id ? "border-primary bg-primary text-primary-foreground" : "border-border bg-card text-muted-foreground hover:border-primary/40 hover:text-foreground"}`}>
                {label}<span className={`rounded-full px-1.5 text-xs ${filter === id ? "bg-primary-foreground/20" : "bg-muted"}`}>{n}</span>
              </button>
            ))}
          </div>
        </div>
      )}
      {history.length === 0 ? (
        <EmptyState icon={History} title="Your history is empty" body="Scans you analyze are saved here automatically, so you can look back at how you sorted each item."
          action={<Button onClick={onGoScan} className="glow h-11 rounded-xl"><ScanLine className="size-4" /> Scan Waste</Button>} />
      ) : shown.length === 0 ? (
        <div className="mt-5"><EmptyState icon={Search} title="No matches" body="No scans match your search or filter."
          action={<Button variant="outline" className="h-11 rounded-xl" onClick={() => { setQuery(""); setFilter("all"); }}>Clear filters</Button>} /></div>
      ) : (
        <div className="mt-5 grid gap-4 sm:grid-cols-2 sm:gap-5 xl:grid-cols-3">
          {shown.map((h) => (
            <article key={h.id} className="glass lift flex min-w-0 flex-col overflow-hidden rounded-2xl">
              <div className="relative bg-muted">
                <img src={h.image} alt={h.item} loading="lazy" className="aspect-video w-full object-cover" />
                <div className="absolute left-3 top-3"><CategoryBadge c={h.category} compact /></div>
                <Button variant="secondary" size="icon" aria-label={`Delete scan of ${h.item}`}
                  className="absolute right-3 top-3 size-10 rounded-full text-destructive shadow-md hover:text-destructive" onClick={() => onDelete(h.id)}>
                  <Trash2 className="size-4" />
                </Button>
              </div>
              <div className="flex flex-1 flex-col p-4 sm:p-5">
                <h3 className="font-semibold">{h.item}</h3>
                <p className="text-sm font-medium text-primary">{h.bin}</p>
                <ul className="mt-2 space-y-1 text-sm text-muted-foreground">
                  {h.steps.slice(0, 3).map((s, i) => <li key={i} className="flex gap-2"><CheckCircle2 className="mt-0.5 size-3.5 shrink-0 text-primary" />{s}</li>)}
                </ul>
                <div className="mt-auto flex items-center justify-between gap-2 border-t border-border pt-3 text-xs text-muted-foreground">
                  <span className="flex items-center gap-1"><Clock className="size-3" />{fmtDate(h.createdAt)}</span>
                  <Button variant="outline" size="sm" className="h-9 rounded-lg" onClick={() => onView(h)}><Eye className="size-4" /> View</Button>
                </div>
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}

/* ---------------- Learn ---------------- */
function Learn() {
  const categories = [
    { tone: "bg-bin-green", icon: Leaf, title: "Organic", sub: "Wet waste", bin: "Green Bin", items: ["Food scraps", "Fruit & vegetable peels", "Tea leaves & flowers", "Garden and plant waste"], tip: "Keep wet waste out of plastic bags — bagged peels can't compost." },
    { tone: "bg-bin-blue", icon: Recycle, title: "Recyclable", sub: "Dry recyclable waste", bin: "Blue Bin", items: ["Clean plastic bottles", "Paper & newspapers", "Cardboard boxes", "Rinsed cans & glass"], tip: "Rinse containers and flatten cardboard before disposal." },
    { tone: "bg-bin-red", icon: AlertTriangle, title: "Hazardous", sub: "Household hazardous waste", bin: "Red Bin (hazardous depot)", items: ["Batteries", "Aerosol spray cans", "Paint & chemicals", "Expired medicines"], tip: "Never mix with regular waste — hand it over at a collection centre." },
    { tone: "bg-bin-red", icon: Cpu, title: "E-Waste", sub: "Electronic waste", bin: "Red Bin (special e-waste bin)", items: ["Phones & tablets", "Cables & chargers", "Circuit boards", "Small appliances"], tip: "Wipe personal data first, then use a certified e-waste recycler." },
    { tone: "bg-bin-black", icon: Trash2, title: "Residual", sub: "Non-recyclable waste", bin: "Black Bin", items: ["Chip & biscuit wrappers", "Multi-layer laminates", "Soiled tissues & wipes", "Broken ceramics"], tip: "If it can't be recycled or composted, it belongs here." },
  ];
  return (
    <section>
      <PageHeader title="Learn Hub" sub="Segregation rules — what goes in each bin, and why." />
      <div className="grid gap-4 sm:grid-cols-2 sm:gap-5 xl:grid-cols-3">
        {categories.map(({ tone, icon: Icon, title, sub, bin, items, tip }) => (
          <article key={title} className="glass lift flex min-w-0 flex-col overflow-hidden rounded-2xl">
            <div className={`${tone} flex items-center gap-4 p-5 text-primary-foreground`}>
              <span className="grid size-12 shrink-0 place-items-center rounded-2xl bg-card/20"><Icon className="size-6" /></span>
              <div className="min-w-0"><h3 className="text-lg font-bold">{title}</h3><p className="text-sm opacity-90">{sub}</p></div>
            </div>
            <div className="flex flex-1 flex-col p-5">
              <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Examples</p>
              <ul className="mt-2 grid gap-2">
                {items.map((i) => <li key={i} className="flex items-center gap-2 rounded-lg bg-muted px-3 py-2 text-sm"><CheckCircle2 className="size-4 shrink-0 text-primary" />{i}</li>)}
              </ul>
              <p className="mt-4 flex items-start gap-2 rounded-xl bg-secondary p-3 text-sm text-secondary-foreground"><Sparkles className="mt-0.5 size-4 shrink-0 text-primary" />{tip}</p>
              <p className="mt-auto pt-4 text-sm font-semibold">Goes to: <span className="text-primary">{bin}</span></p>
            </div>
          </article>
        ))}
        <article className="flex min-w-0 flex-col overflow-hidden rounded-2xl border-2 border-dashed border-warning bg-card">
          <div className="bg-warning-soft flex items-center gap-4 p-5 text-warning-foreground">
            <span className="grid size-12 shrink-0 place-items-center rounded-2xl bg-card"><HelpCircle className="size-6" /></span>
            <div className="min-w-0"><h3 className="text-lg font-bold">Uncertain</h3><p className="text-sm opacity-90">When the scanner can't decide</p></div>
          </div>
          <div className="flex flex-1 flex-col p-5">
            <p className="text-sm">If the photo is blurry, shows several mixed items, or the model's confidence falls below 55%, EcoLens reports <strong>Uncertain</strong> and recommends a rescan instead of guessing a bin.</p>
            <p className="mt-4 flex items-start gap-2 rounded-xl bg-secondary p-3 text-sm text-secondary-foreground"><Sparkles className="mt-0.5 size-4 shrink-0 text-primary" />One clear item per photo gives the best result.</p>
            <p className="mt-auto pt-4 text-sm font-semibold">Action: <span className="text-primary">retake the photo</span></p>
          </div>
        </article>
      </div>
    </section>
  );
}
