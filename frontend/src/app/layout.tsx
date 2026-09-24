import type { Metadata } from "next";
import { IBM_Plex_Sans, Source_Serif_4 } from "next/font/google";
import "./globals.css";

const plexSans = IBM_Plex_Sans({
  variable: "--font-plex-sans",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

const sourceSerif = Source_Serif_4({
  variable: "--font-source-serif",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "SciChat: ask a paper, get cited answers",
  description: "Ask questions about scientific papers and get answers grounded in, and cited to, the paper's own text.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${plexSans.variable} ${sourceSerif.variable} font-sans antialiased bg-background text-foreground h-screen overflow-hidden`}>
        {children}
      </body>
    </html>
  );
}
