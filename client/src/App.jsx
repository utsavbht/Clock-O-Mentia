import React from "react";
import Description from "./components/Description";
import Footer from "./components/Footer";
import Hero from "./components/Hero";
import History from "./components/FileUpload";
import Solution from "./components/Solution";

export default function App() {
  return (
    <>
      <Hero />
      <Description />
      <Solution />
      <History />
      <Footer />
    </>
  );
}
