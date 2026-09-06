import { motion } from 'framer-motion';
import { useFileStore } from '@/stores/file-store';
import { fadeUp } from '@/lib/motion';

export function MemoryWarning() {
  const isLargeFile = useFileStore((s) => s.isLargeFile);
  const file = useFileStore((s) => s.file);

  if (!isLargeFile || !file) return null;

  const sizeMB = (file.size / (1024 * 1024)).toFixed(0);

  return (
    <motion.div
      variants={fadeUp}
      initial="hidden"
      animate="visible"
      className="rounded-[14px] p-3 text-xs border border-warn/20 bg-warn/5 text-warn flex items-start gap-2"
    >
      <svg
        className="w-4 h-4 shrink-0 mt-0.5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1.5}
          d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"
        />
      </svg>
      <div>
        <p className="font-medium">Large file detected ({sizeMB} MB)</p>
        <p className="mt-0.5 text-ink-soft">
          Processing may be slow and could cause memory issues. Consider trimming the video first.
        </p>
      </div>
    </motion.div>
  );
}
