use proc_macro::{TokenStream};
use proc_macro2::{TokenStream as TokenStream2, TokenTree as TokenTree2, Ident as Ident2};
use quote::{quote, ToTokens};
use syn::{parse_macro_input, Ident};
use syn::spanned::Spanned;

/// Clones a token tree with specified identifiers replaced. Replace identifiers by specifying their
/// name as an attribute equal to another identifier name.
///
/// This attribute can be stacked to create multiple clones.
///
/// # Warning
/// If more than one instance of this macro is called, make sure to rename the main identifier / name of
/// whatever you call this macro on. Otherwise, a "declared multiple times" error will appear.
///
/// # Example
/// The following function:
/// ```
/// #[replace_idents(axis_stuff = axis_stuff1, axis = z, axis1 = x, axis2 = y)]
/// #[replace_idents(axis_stuff = axis_stuff2, axis = z, axis1 = y, axis2 = x)]
/// fn axis_stuff(b: UVec3) -> f32 {
///     b.axis * (b.axis2 - b.axis1)
/// }
/// ```
/// turns into:
///
/// ```
/// #[replace_idents(axis_stuff = axis_stuff2, axis = z, axis1 = y, axis2 = x)]
/// fn axis_stuff(b: UVec3) -> f32 {
///     b.axis * (b.axis2 - b.axis1)
/// }
///
/// fn axis_stuff1(b: UVec3) -> f32 {
///     b.z * (b.y - b.x)
/// }
/// ```
#[proc_macro_attribute]
pub fn replace_idents(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut targets = vec![];
    let mut replacements = vec![];
    let attr_parser = syn::meta::parser(|meta| {
        if let Some(ident) = meta.path.get_ident() {
            targets.push(ident.to_string());
            let replacement_ident: Ident = meta.value()?.parse()?;
            replacements.push(replacement_ident.to_string());
        } else {
            return Err(meta.error(
                format!("Unsupported attribute path: {}", meta.path.clone().into_token_stream())
            ));
        }
        Ok(())
    });
    parse_macro_input!(attr with attr_parser);

    let item2 = TokenStream2::from(item.clone());

    if targets.is_empty() || replacements.is_empty() {
        return syn::Error::new(item2.span(), "Expected at least one identifier to replace")
            .to_compile_error().into();
    }

    let mut new_stream = replace_idents_in_stream(
        item2.clone(),
        targets[0].as_str(),
        replacements[0].as_str()
    );
    for (target, replacement) in targets
        .iter()
        .zip(replacements.iter())
        .skip(1)
    {
        new_stream = replace_idents_in_stream(new_stream, target.as_str(), replacement.as_str());
    }
    let new_stream = TokenStream::from(new_stream);

    let mut item_clone = parse_macro_input!(new_stream as syn::Item);
    
    let attrs = match &mut item_clone {
        syn::Item::Struct(s) => Some(&mut s.attrs),
        syn::Item::Fn(f) => Some(&mut f.attrs),
        syn::Item::Enum(e) => Some(&mut e.attrs),
        syn::Item::Impl(i) => Some(&mut i.attrs),
        _ => None,
    };

    // Remove all other instances of this macro from clone
    let is_this_attr = |a: &syn::Attribute| {
        a.path().is_ident("replace_idents") || a.path().is_ident("taser_em_macros::replace_idents")
    };
    let Some(attrs) = attrs else {
        return syn::Error::new(item2.span(), "Only Structs, Functions, Impls, and Enums are supported")
            .to_compile_error().into();
    };
    attrs.retain(|a| !is_this_attr(a));

    let item_parsed = parse_macro_input!(item as syn::Item);
    let original_attrs = match &item_parsed {
        syn::Item::Struct(s) => &s.attrs,
        syn::Item::Fn(f) => &f.attrs,
        syn::Item::Enum(e) => &e.attrs,
        syn::Item::Impl(i) => &i.attrs,
        _ => unreachable!(),
    };
    let n_self_attrs = original_attrs.iter().filter(|a| is_this_attr(a)).count();
    if n_self_attrs != 0 { // if more instances of this macro are left, just make a clone of the item
        TokenStream::from(quote! {
            #item2
            #item_clone
        })
    }
    else { // otherwise, overwrite the item
        TokenStream::from(quote! {
            #item_clone
        })
    }
}

fn replace_idents_in_stream(stream: TokenStream2, target: &str, replacement: &str) -> TokenStream2 {
    let mut new_stream = TokenStream2::new();

    for tt in stream {
        match tt {
            TokenTree2::Ident(ref ident) if ident == target => {
                let new_ident = Ident2::new(replacement, ident.span());
                new_stream.extend(quote! { #new_ident });
            }
            TokenTree2::Group(group) => {
                let inner_stream = replace_idents_in_stream(group.stream(), target, replacement);
                let mut new_group = proc_macro2::Group::new(group.delimiter(), inner_stream);
                new_group.set_span(group.span());
                new_stream.extend(quote! { #new_group });
            }
            other => new_stream.extend(quote! { #other })
        }
    }

    new_stream
}