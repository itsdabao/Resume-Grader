import type { Candidate } from "./components/leaderboard-card";
import type { Commentary } from "./components/commentary-panel";
import type { CVData } from "./components/cv-preview";

export const SAMPLE_CANDIDATES: Candidate[] = [
  { id: "1", name: "Amara Okafor", role: "Senior Frontend Engineer", score: 94 },
  { id: "2", name: "Daniel Reyes", role: "Full-Stack Developer", score: 88 },
  { id: "3", name: "Priya Natarajan", role: "Product Engineer", score: 82 },
  { id: "4", name: "Marcus Lindqvist", role: "Frontend Engineer", score: 76 },
  { id: "5", name: "Yuki Tanaka", role: "UI Engineer", score: 71 },
  { id: "6", name: "Sofia Marchetti", role: "Junior Developer", score: 64 },
  { id: "7", name: "Jordan Bell", role: "Web Developer", score: 58 },
];

export const COMMENTARIES: Record<string, Commentary> = {
  "1": {
    score: 94,
    strengths: [
      "Led design system at scale across 4 product teams",
      "Strong TypeScript & React performance expertise",
      "Open-source contributions to popular tooling",
    ],
    weaknesses: [
      "Limited backend/infra exposure",
      "No formal management experience listed",
    ],
    reasoning:
      "Amara presents an exceptionally strong frontend profile with quantified impact (40% reduction in bundle size, 2.1s LCP improvement). Their design system leadership directly maps to the role's scope. Resume structure is clean, achievements are metric-driven, and skills cluster aligns with our stack (React, TS, Vite, Tailwind). Minor concerns: backend depth is shallow which may matter for cross-functional collaboration on the platform team.",
  },
  "2": {
    score: 88,
    strengths: [
      "Balanced full-stack background (Node + React)",
      "Shipped 3 products from 0→1",
      "Mentorship experience with juniors",
    ],
    weaknesses: [
      "Generic summary lacks role-specific keywords",
      "Job tenure averages under 18 months",
    ],
    reasoning:
      "Daniel is a versatile candidate with clear product instincts. The 0→1 experience is compelling but short tenures raise retention questions worth probing in interview. Technical breadth is solid; depth on the frontend is slightly below Amara's level.",
  },
  "3": {
    score: 82,
    strengths: [
      "Strong product thinking and user research background",
      "Comfortable with experimentation frameworks",
    ],
    weaknesses: [
      "Less senior-level architectural experience",
      "Buzzword-heavy in places without backing metrics",
    ],
    reasoning:
      "Priya is a product-minded engineer well-suited to feature teams. Resume leans more on outcomes than on technical specifics, which works for product engineering roles but reduces evaluator confidence on system design questions.",
  },
};

export const CV_DATA: Record<string, CVData> = {
  "1": {
    name: "Amara Okafor",
    title: "Senior Frontend Engineer",
    email: "amara.okafor@example.com",
    location: "Berlin, Germany",
    summary: [
      { text: "Senior frontend engineer with " },
      {
        text: "8+ years building design systems",
        annotation: {
          kind: "bonus",
          tip: "Direct match to job requirement: design system leadership.",
        },
      },
      { text: " used by " },
      {
        text: "4 product teams and 30+ engineers",
        annotation: {
          kind: "bonus",
          tip: "Quantified scope shows real organizational impact.",
        },
      },
      { text: ". Passionate about web performance and accessibility." },
    ],
    experience: [
      {
        role: "Staff Frontend Engineer",
        company: "Lumen Labs",
        period: "2022 — Present",
        bullets: [
          [
            { text: "Reduced bundle size by " },
            {
              text: "40% through code-splitting and tree-shaking audits",
              annotation: {
                kind: "bonus",
                tip: "Concrete, measurable performance win.",
              },
            },
            { text: "." },
          ],
          [
            { text: "Owned the migration from CRA to Vite, improving CI build times by 3x." },
          ],
          [
            {
              text: "Responsible for various frontend tasks",
              annotation: {
                kind: "penalty",
                tip: "Vague responsibility statement — lacks measurable outcomes.",
              },
            },
            { text: " across the platform." },
          ],
        ],
      },
      {
        role: "Senior Frontend Engineer",
        company: "Helios Health",
        period: "2019 — 2022",
        bullets: [
          [
            { text: "Built the company-wide " },
            {
              text: "design system used in 12 production apps",
              annotation: {
                kind: "bonus",
                tip: "High-leverage work directly relevant to this role.",
              },
            },
            { text: "." },
          ],
          [
            { text: "Mentored 6 junior engineers through structured code reviews." },
          ],
        ],
      },
    ],
    skills: [
      { text: "TypeScript, React, Next.js, " },
      {
        text: "Tailwind, Vite, Radix, Storybook",
        annotation: {
          kind: "bonus",
          tip: "Stack matches our team's tooling almost 1:1.",
        },
      },
      { text: ", Node.js, " },
      {
        text: "jQuery",
        annotation: {
          kind: "penalty",
          tip: "Listing legacy tooling without context can signal dated skills.",
        },
      },
      { text: ", Playwright, Figma." },
    ],
    education: [
      {
        school: "TU Munich",
        degree: "M.Sc. Computer Science",
        year: "2016",
      },
    ],
  },
};

export function buildFallbackCV(c: Candidate): CVData {
  return {
    name: c.name,
    title: c.role,
    email: `${c.name.toLowerCase().replace(/\s+/g, ".")}@example.com`,
    location: "Remote",
    summary: [
      { text: `${c.role} with proven track record. ` },
      {
        text: "Strong communication and collaboration",
        annotation: {
          kind: "bonus",
          tip: "Soft-skill signal verified by reference notes.",
        },
      },
      { text: " across cross-functional teams." },
    ],
    experience: [
      {
        role: c.role,
        company: "Previous Co.",
        period: "2021 — Present",
        bullets: [
          [
            {
              text: "Shipped multiple production features on schedule",
              annotation: {
                kind: "bonus",
                tip: "Demonstrates reliable delivery cadence.",
              },
            },
            { text: "." },
          ],
          [
            {
              text: "Helped out with various things as needed",
              annotation: {
                kind: "penalty",
                tip: "Non-specific; reduces evaluator confidence in impact.",
              },
            },
            { text: "." },
          ],
        ],
      },
    ],
    skills: [{ text: "React, TypeScript, CSS, Git, REST APIs." }],
    education: [
      { school: "State University", degree: "B.Sc. Computer Science", year: "2019" },
    ],
  };
}
