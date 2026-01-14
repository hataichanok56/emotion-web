import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Emotion AI Scanner",
  description: "Real-time Neural Face Analysis",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="th">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
